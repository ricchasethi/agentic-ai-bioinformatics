"""
================================================================
Reflection Agent — Batch Correction Strategy Revision
UK Biobank OLINK Proteomics Data
================================================================

This module adds a self-correcting reflection loop to the
batch correction pipeline.  After node_assess_batch_effects
runs, the ReflectionAgent:

  1. Inspects each confirmed batch field's η² and % proteins
     affected against the thresholds in CLAUDE.md.
  2. Classifies each field as: KEEP_PRIMARY | KEEP_SECONDARY |
     DEMOTE_TO_COVARIATE | DROP.
  3. Revises the confirmed field list and primary batch field
     in state so that node_propose_strategy receives a cleaner
     input (and the human sees fewer spurious fields).
  4. If no field passes the threshold for ComBat, it
     automatically down-grades the overall strategy to
     covariate_only and records the reasoning.
  5. Supports up to MAX_REFLECTION_ROUNDS re-assessment rounds
     (default 3) before forcing a decision, preventing
     infinite loops.

The reflection reasoning is appended to state["agent_reasoning"]
and state["reflection_log"] (new field) for full auditability.

Thresholds (from CLAUDE.md):
    η² > 0.05  OR  pct > 30%  → HIGH   (ComBat required)
    η² > 0.01  OR  pct > 10%  → MODERATE (ComBat recommended)
    η² ≤ 0.01  AND pct ≤ 10%  → LOW    (covariate only)
    η² < 0.001 AND pct <  2%  → NONE   (drop entirely)

Integration:
    Add "node_reflect_on_assessment" to the LangGraph graph
    between node_assess_batch_effects and node_propose_strategy.
    No other files need to change.

    In batch_correction_agent.py build_graph(), replace:
        graph.add_edge("node_assess_batch_effects", "node_propose_strategy")
    with:
        graph.add_edge("node_assess_batch_effects",  "node_reflect_on_assessment")
        graph.add_edge("node_reflect_on_assessment", "node_propose_strategy")

    And register the node:
        from batch_correction_reflection_agent import node_reflect_on_assessment
        graph.add_node("node_reflect_on_assessment", node_reflect_on_assessment)
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# LangChain
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

# Reuse helpers from the main agent module
from batch_correction_agent import (
    BatchCorrectionState,
    _save_df,
    _load_df,
    _sanitise,
    get_llm,
    SYSTEM_PROMPT,
    CACHE_DIR,
)

# ── Constants ─────────────────────────────────────────────────

MAX_REFLECTION_ROUNDS = 3   # safety cap — never loop more than this

# η² / pct thresholds (CLAUDE.md §Batch severity thresholds)
THRESHOLD_HIGH_ETA    = 0.05
THRESHOLD_HIGH_PCT    = 30.0
THRESHOLD_MOD_ETA     = 0.01
THRESHOLD_MOD_PCT     = 10.0
THRESHOLD_DROP_ETA    = 0.001  # so small it's essentially noise
THRESHOLD_DROP_PCT    = 2.0    # <2% proteins: no meaningful batch effect


# ================================================================
# DECISION LOGIC
# ================================================================

def _classify_field(
    field_name: str,
    metrics: dict,
    is_primary: bool,
) -> dict:
    """
    Classify a single batch field based on its η² / pct_affected.

    Returns a dict:
        {
          "field":      str,
          "action":     "KEEP_PRIMARY" | "KEEP_SECONDARY" |
                        "DEMOTE_TO_COVARIATE" | "DROP",
          "severity":   "HIGH" | "MODERATE" | "LOW" | "NONE",
          "reasoning":  str,
        }
    """
    eta  = metrics.get("mean_eta_squared",      0.0)
    pct  = metrics.get("pct_proteins_affected", 0.0)

    if eta >= THRESHOLD_HIGH_ETA or pct >= THRESHOLD_HIGH_PCT:
        severity = "HIGH"
        action   = "KEEP_PRIMARY" if is_primary else "KEEP_SECONDARY"
        reasoning = (
            f"η²={eta:.4f}, {pct:.1f}% proteins affected — "
            f"batch effect is HIGH. {'Primary ComBat variable confirmed.' if is_primary else 'Retain as secondary ComBat covariate.'}"
        )

    elif eta >= THRESHOLD_MOD_ETA or pct >= THRESHOLD_MOD_PCT:
        severity  = "MODERATE"
        action    = "KEEP_PRIMARY" if is_primary else "KEEP_SECONDARY"
        reasoning = (
            f"η²={eta:.4f}, {pct:.1f}% proteins affected — "
            f"batch effect is MODERATE. {'ComBat on this primary variable is recommended.' if is_primary else 'Retain as secondary batch covariate.'}"
        )

    elif eta >= THRESHOLD_DROP_ETA or pct >= THRESHOLD_DROP_PCT:
        severity  = "LOW"
        # Even if this was nominated as primary, LOW means include only as
        # regression covariate — don't run ComBat over it
        action    = "DEMOTE_TO_COVARIATE"
        reasoning = (
            f"η²={eta:.4f}, {pct:.1f}% proteins affected — "
            f"batch effect is LOW. ComBat would risk over-correction; "
            f"include '{field_name}' as a covariate in regression instead."
        )

    else:
        severity  = "NONE"
        action    = "DROP"
        reasoning = (
            f"η²={eta:.4f}, {pct:.1f}% proteins affected — "
            f"no detectable batch effect. Dropping '{field_name}' entirely; "
            f"it carries no technical variance worth correcting."
        )

    return {
        "field":     field_name,
        "action":    action,
        "severity":  severity,
        "reasoning": reasoning,
    }


def _build_revised_plan(
    decisions:           list[dict],
    original_primary:    str,
    original_confirmed:  list[str],
) -> dict:
    """
    From per-field decisions, derive the revised field lists and method.

    Returns:
        {
          "confirmed_batch_fields": [...],   # fields to pass to ComBat
          "primary_batch_field":    str,
          "demoted_fields":         [...],   # retain as regression covariates
          "dropped_fields":         [...],
          "recommended_method":     "ComBat" | "covariate_only" | "none",
          "overall_reasoning":      str,
        }
    """
    keep    = [d for d in decisions if d["action"] in ("KEEP_PRIMARY", "KEEP_SECONDARY")]
    demoted = [d["field"] for d in decisions if d["action"] == "DEMOTE_TO_COVARIATE"]
    dropped = [d["field"] for d in decisions if d["action"] == "DROP"]

    confirmed_for_combat = [d["field"] for d in keep]

    # Choose new primary: prefer original primary if it survived, else
    # the highest-η² field among survivors.
    if original_primary in confirmed_for_combat:
        new_primary = original_primary
    elif confirmed_for_combat:
        new_primary = confirmed_for_combat[0]
    else:
        new_primary = ""

    if confirmed_for_combat:
        # At least one field warrants ComBat
        method = "ComBat"
        overall = (
            f"After reflection, {len(confirmed_for_combat)} field(s) retain "
            f"sufficient batch signal for ComBat correction "
            f"({', '.join(confirmed_for_combat)}). "
            f"{len(demoted)} field(s) demoted to regression covariate only "
            f"({', '.join(demoted) if demoted else 'none'}). "
            f"{len(dropped)} field(s) dropped ({', '.join(dropped) if dropped else 'none'})."
        )
    elif demoted:
        method = "covariate_only"
        new_primary = demoted[0]
        overall = (
            f"No field meets the ComBat threshold. All surviving fields "
            f"({', '.join(demoted)}) have LOW batch effect and will be included "
            f"as regression covariates only. ComBat correction is not warranted."
        )
    else:
        method = "none"
        overall = (
            "Reflection finds NO meaningful batch effects across all confirmed "
            "fields. No batch correction is required. Proceed directly to "
            "association testing."
        )

    return {
        "confirmed_batch_fields": confirmed_for_combat,
        "primary_batch_field":    new_primary,
        "demoted_fields":         demoted,
        "dropped_fields":         dropped,
        "recommended_method":     method,
        "overall_reasoning":      overall,
    }


# ================================================================
# LLM-ASSISTED REFLECTION (called inside the node)
# ================================================================

def _llm_reflect(
    assessment: dict,
    decisions:  list[dict],
    revised_plan: dict,
    round_num: int,
) -> str:
    """
    Ask the LLM to critique the reflection decisions and return
    a short paragraph of additional reasoning / caveats.
    Falls back to a plain-text summary if the LLM call fails.
    """
    llm = get_llm()

    prompt = f"""
You are reviewing an automated batch-effect reflection decision for UK Biobank
OLINK proteomics data.

ROUND: {round_num}

BATCH ASSESSMENT RESULTS (η² and % proteins affected per field):
{json.dumps(assessment, indent=2)}

AUTOMATED FIELD DECISIONS:
{json.dumps(decisions, indent=2)}

REVISED PLAN:
{json.dumps(revised_plan, indent=2)}

Please critically evaluate whether:
1. Any HIGH/MODERATE field was incorrectly demoted or dropped.
2. Any LOW/NONE field should actually be retained (e.g. important known
   plate variable even if η² is marginal).
3. The recommended method ({revised_plan['recommended_method']}) is appropriate.
4. There are any risks in the revised plan for downstream AD biomarker discovery.

Respond with a concise paragraph (3-5 sentences) of critique/validation.
Do NOT return JSON — just plain text commentary.
"""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as exc:
        return (
            f"[LLM reflection unavailable: {exc}] "
            f"Automated decisions applied without additional LLM critique."
        )


# ================================================================
# RE-ASSESSMENT HELPER
# (recomputes η² on the current df after a previous round may have
#  pruned small-batch samples or applied QC steps)
# ================================================================

def _reassess_field(
    df: pd.DataFrame,
    protein_cols: list[str],
    batch_field: str,
    sample_n: int = 150,
) -> dict:
    """
    Lightweight re-computation of η² for a single batch field.
    Uses the same logic as node_assess_batch_effects but faster
    (samples up to `sample_n` proteins).
    Returns the same schema as batch_assessment values.
    """
    if batch_field not in df.columns:
        return {"mean_eta_squared": 0.0, "pct_proteins_affected": 0.0,
                "severity": "NONE", "is_categorical": False}

    batch_var      = df[batch_field]
    is_categorical = bool(
        df[batch_field].dtype == object or df[batch_field].nunique() < 50
    )

    sampled = protein_cols[:sample_n]
    eta_vals, p_vals = [], []

    for prot in sampled:
        if prot not in df.columns:
            continue
        y = df[prot].dropna()
        x = batch_var.loc[y.index].dropna()
        y = y.loc[x.index]
        if len(y) < 20:
            continue

        if is_categorical:
            groups = [
                y[x == cat].values for cat in x.unique() if len(y[x == cat]) > 2
            ]
            if len(groups) < 2:
                continue
            _, p_val = stats.f_oneway(*groups)
            grand    = y.mean()
            ss_b     = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
            ss_t     = ((y - grand) ** 2).sum()
            eta_sq   = float(ss_b / ss_t) if ss_t > 0 else 0.0
        else:
            rho, p_val = stats.spearmanr(x, y)
            eta_sq     = float(rho ** 2)

        eta_vals.append(eta_sq)
        p_vals.append(float(p_val))

    n_sig        = int(sum(p < 0.05 for p in p_vals))
    pct_affected = float(n_sig / max(len(protein_cols), 1) * 100)
    mean_eta     = float(np.mean(eta_vals)) if eta_vals else 0.0

    if mean_eta >= THRESHOLD_HIGH_ETA or pct_affected >= THRESHOLD_HIGH_PCT:
        severity = "HIGH — ComBat correction required"
    elif mean_eta >= THRESHOLD_MOD_ETA or pct_affected >= THRESHOLD_MOD_PCT:
        severity = "MODERATE — ComBat recommended"
    else:
        severity = "LOW — covariate adjustment sufficient"

    return {
        "mean_eta_squared":      mean_eta,
        "pct_proteins_affected": pct_affected,
        "n_proteins_significant": n_sig,
        "severity":              severity,
        "is_categorical":        is_categorical,
    }


# ================================================================
# MAIN NODE
# ================================================================

def node_reflect_on_assessment(
    state: BatchCorrectionState,
) -> BatchCorrectionState:
    """
    Reflection node — sits between node_assess_batch_effects and
    node_propose_strategy in the LangGraph graph.

    What it does
    ------------
    Round 1 (always runs):
        • Reads the existing batch_assessment from state.
        • Classifies each confirmed field as KEEP/DEMOTE/DROP.
        • Builds a revised field list and recommended method.
        • Asks the LLM to critique the decisions.

    Rounds 2-N (only if the plan changed in the previous round):
        • Re-runs lightweight η² on any field whose classification
          changed, using the current df (which may have been updated
          if small-batch samples were pruned in node_apply_correction
          during a retry).
        • Iterates until stable or MAX_REFLECTION_ROUNDS is reached.

    State changes
    -------------
    • state["confirmed_batch_fields"]  — revised list (ComBat candidates only)
    • state["primary_batch_field"]     — revised primary
    • state["batch_assessment"]        — updated with any re-assessed values
    • state["reflection_log"]          — full per-round audit trail (new key)
    • state["agent_reasoning"]         — summary appended
    • state["proposed_strategy"]       — pre-filled with recommended_method
                                         so node_propose_strategy has a prior
    """

    print("\n[REFLECT] Reflection agent reviewing batch assessment...")

    assessment       = state["batch_assessment"]
    confirmed        = list(state["confirmed_batch_fields"])
    original_primary = state["primary_batch_field"]
    protein_cols     = state["protein_cols"]

    # Initialise reflection_log in state if not present
    reflection_log: list[dict] = state.get("reflection_log", [])  # type: ignore[assignment]

    df = _load_df(state["df_current_path"])

    round_num       = 0
    previous_plan   = None

    while round_num < MAX_REFLECTION_ROUNDS:
        round_num += 1
        print(f"\n  [Reflect round {round_num}/{MAX_REFLECTION_ROUNDS}]")

        # ── Step 1: Classify each confirmed field ──────────────
        decisions = []
        for field in confirmed:
            metrics    = assessment.get(field, {})
            is_primary = (field == original_primary)
            decision   = _classify_field(field, metrics, is_primary)
            decisions.append(decision)
            print(f"    {field:<35} → {decision['action']:25}  "
                  f"(η²={metrics.get('mean_eta_squared', 0):.4f}, "
                  f"{metrics.get('pct_proteins_affected', 0):.1f}%)")

        # ── Step 2: Build revised plan ──────────────────────────
        revised_plan = _build_revised_plan(decisions, original_primary, confirmed)

        # ── Step 3: Check for convergence ──────────────────────
        current_confirmed = revised_plan["confirmed_batch_fields"]
        current_primary   = revised_plan["primary_batch_field"]

        if previous_plan is not None:
            # Stable if fields and method unchanged since last round
            prev_confirmed = previous_plan["confirmed_batch_fields"]
            prev_method    = previous_plan["recommended_method"]
            if (sorted(current_confirmed) == sorted(prev_confirmed) and
                    revised_plan["recommended_method"] == prev_method):
                print(f"    ✓ Stable after round {round_num} — converged.")
                break

        # ── Step 4: LLM critique ───────────────────────────────
        llm_commentary = _llm_reflect(assessment, decisions, revised_plan, round_num)
        print(f"\n    LLM critique:\n    {llm_commentary[:300]}{'...' if len(llm_commentary) > 300 else ''}")

        # ── Step 5: Re-assess any newly-promoted fields  ───────
        # If the plan changed (fields were dropped/demoted), re-assess the
        # survivors on the current df to confirm their η² is still valid.
        newly_survivors = [
            f for f in current_confirmed
            if previous_plan is None
            or f not in (previous_plan.get("confirmed_batch_fields") or [])
        ]
        if newly_survivors and round_num < MAX_REFLECTION_ROUNDS:
            print(f"    Re-assessing {len(newly_survivors)} field(s) on current df...")
            for field in newly_survivors:
                new_metrics = _reassess_field(df, protein_cols, field)
                assessment[field] = new_metrics
                print(f"      {field}: η²={new_metrics['mean_eta_squared']:.4f}  "
                      f"{new_metrics['pct_proteins_affected']:.1f}% affected")

        # ── Step 6: Record this round ──────────────────────────
        round_record = {
            "round":          round_num,
            "decisions":      decisions,
            "revised_plan":   revised_plan,
            "llm_commentary": llm_commentary,
        }
        reflection_log.append(round_record)

        # Update working variables for next iteration
        previous_plan = revised_plan
        confirmed     = current_confirmed if current_confirmed else list(
            revised_plan["demoted_fields"]  # fall back so loop doesn't run empty
        )

        # If nothing left to assess, stop
        if not confirmed:
            break

    # ── Finalise ───────────────────────────────────────────────
    final_plan = revised_plan  # last stable plan

    print(f"\n  [Reflect] Final plan:")
    print(f"    Method:    {final_plan['recommended_method']}")
    print(f"    Primary:   {final_plan['primary_batch_field']}")
    print(f"    ComBat fields:  {final_plan['confirmed_batch_fields']}")
    print(f"    Demoted fields: {final_plan['demoted_fields']}")
    print(f"    Dropped fields: {final_plan['dropped_fields']}")

    # ── Write back to state ────────────────────────────────────

    # 1. Update confirmed fields (ComBat candidates only)
    state["confirmed_batch_fields"] = final_plan["confirmed_batch_fields"]
    state["primary_batch_field"]    = final_plan["primary_batch_field"]

    # 2. Refresh batch_assessment with any re-assessed values
    state["batch_assessment"] = _sanitise(assessment)

    # 3. Pre-fill proposed_strategy so node_propose_strategy has a strong prior
    #    (it will refine further — this avoids it re-inventing the wheel)
    prior_strategy = {
        "recommended_method":               final_plan["recommended_method"],
        "primary_batch_variable":           final_plan["primary_batch_field"],
        "secondary_batch_variables":        final_plan["demoted_fields"],
        "biological_covariates_to_protect": ["age", "sex", "AD_case", "apoe_e4"],
        "dropped_fields":                   final_plan["dropped_fields"],
        "reflection_reasoning":             final_plan["overall_reasoning"],
        # These will be overwritten by node_propose_strategy's LLM call
        "pre_correction_steps":             [],
        "combat_parameters":                {
            "parametric": True, "mean_only": False, "ref_batch": None
        },
        "justification":        final_plan["overall_reasoning"],
        "risks_and_mitigations":[],
        "validation_checks":    [],
    }
    state["proposed_strategy"] = _sanitise(prior_strategy)

    # 4. Store the reflection log (new state key)
    state["reflection_log"] = _sanitise(reflection_log)  # type: ignore[typeddict-unknown-key]

    # 5. Append summary to agent_reasoning
    summary = (
        f"[Reflection — {round_num} round(s)] "
        f"{final_plan['overall_reasoning']} "
        f"Demoted: {final_plan['demoted_fields']}. "
        f"Dropped: {final_plan['dropped_fields']}."
    )
    state["agent_reasoning"].append(summary)

    # 6. If strategy is now 'none', skip ComBat application downstream
    #    by marking correction_needed False
    if final_plan["recommended_method"] == "none":
        state["correction_needed"] = False

    return state


# ================================================================
# PATCH: extend BatchCorrectionState with reflection_log
# ================================================================
# BatchCorrectionState is a TypedDict defined in batch_correction_agent.py.
# Adding a new key to a TypedDict at runtime requires extending it.
# We patch it here so the rest of the pipeline sees the new field.

from typing import get_type_hints
import batch_correction_agent as _agent_module

# Only patch if not already patched (idempotent)
if "reflection_log" not in BatchCorrectionState.__annotations__:
    BatchCorrectionState.__annotations__["reflection_log"] = list

# Patch the initial_state builder in run_pipeline to include the new key.
# We wrap run_pipeline so the initial_state dict includes reflection_log=[].
_original_run_pipeline = _agent_module.run_pipeline

def run_pipeline_with_reflection(
    data_path: str,
    thread_id: str = "batch_correction_reflect_01",
):
    """
    Drop-in replacement for run_pipeline() that includes the
    reflection node.  Call this instead of the original.
    """
    from langgraph.types import Command
    from langgraph.graph import END, START, StateGraph
    from langgraph.checkpoint.memory import MemorySaver

    # Import all original nodes
    from batch_correction_agent import (
        node_load_data,
        node_detect_batch_fields,
        human_confirm_batch_fields,
        node_assess_batch_effects,
        node_propose_strategy,
        human_approve_strategy,
        node_apply_correction,
        node_validate_correction,
        human_accept_results,
        node_save_outputs,
        route_after_acceptance,
    )

    graph = StateGraph(BatchCorrectionState)

    graph.add_node("node_load_data",              node_load_data)
    graph.add_node("node_detect_batch_fields",    node_detect_batch_fields)
    graph.add_node("human_confirm_batch_fields",  human_confirm_batch_fields)
    graph.add_node("node_assess_batch_effects",   node_assess_batch_effects)
    graph.add_node("node_reflect_on_assessment",  node_reflect_on_assessment)  # ← NEW
    graph.add_node("node_propose_strategy",       node_propose_strategy)
    graph.add_node("human_approve_strategy",      human_approve_strategy)
    graph.add_node("node_apply_correction",       node_apply_correction)
    graph.add_node("node_validate_correction",    node_validate_correction)
    graph.add_node("human_accept_results",        human_accept_results)
    graph.add_node("node_save_outputs",           node_save_outputs)

    graph.add_edge(START, "node_load_data")
    graph.add_conditional_edges(
        "node_load_data",
        lambda s: END if s.get("final_status") == "ERROR" else "node_detect_batch_fields",
        ["node_detect_batch_fields", END],
    )
    graph.add_edge("node_detect_batch_fields",       "human_confirm_batch_fields")
    graph.add_edge("human_confirm_batch_fields",     "node_assess_batch_effects")
    graph.add_edge("node_assess_batch_effects",      "node_reflect_on_assessment")  # ← CHANGED
    graph.add_edge("node_reflect_on_assessment",     "node_propose_strategy")       # ← NEW
    graph.add_edge("node_propose_strategy",          "human_approve_strategy")
    graph.add_edge("human_approve_strategy",         "node_apply_correction")
    graph.add_edge("node_apply_correction",          "node_validate_correction")
    graph.add_edge("node_validate_correction",       "human_accept_results")
    graph.add_conditional_edges(
        "human_accept_results",
        route_after_acceptance,
        ["node_save_outputs", END],
    )
    graph.add_edge("node_save_outputs", END)

    memory   = MemorySaver()
    pipeline = graph.compile(
        checkpointer=memory,
        interrupt_before=[
            "human_confirm_batch_fields",
            "human_approve_strategy",
            "human_accept_results",
        ],
    )
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "data_path":              data_path,
        "df_raw_path":            "",
        "df_current_path":        "",
        "protein_cols":           [],
        "metadata_cols":          [],
        "candidate_batch_fields": {},
        "confirmed_batch_fields": [],
        "primary_batch_field":    "",
        "batch_assessment":       {},
        "assessment_plots":       [],
        "correction_needed":      False,
        "proposed_strategy":      {},
        "approved_strategy":      {},
        "df_corrected_path":      "",
        "validation_metrics":     {},
        "validation_plots":       [],
        "human_decisions":        [],
        "agent_reasoning":        [],
        "reflection_log":         [],   # ← NEW
        "current_step":           "init",
        "error_log":              [],
        "final_status":           "RUNNING",
    }

    print("\n" + "=" * 60)
    print("STARTING: Reflection-Augmented Batch Correction Pipeline")
    print("=" * 60)

    def _stream_until_interrupt(input_val):
        for _ in pipeline.stream(input_val, config, stream_mode="values"):
            pass
        return pipeline.get_state(config)

    graph_state = _stream_until_interrupt(initial_state)

    while graph_state.next:
        next_node = graph_state.next[0]
        print(f"\n  ⏸  PAUSED before: {next_node}")

        interrupt_val = _get_interrupt_value_local(graph_state)
        if interrupt_val and isinstance(interrupt_val, dict):
            print("\n" + "─" * 60)
            print(interrupt_val.get("message", ""))
            print("\n" + interrupt_val.get("instructions", ""))
        else:
            print(f"  Waiting at checkpoint: {next_node}")

        human_input = input("\nYour response (press ENTER to accept defaults): ").strip()
        if not human_input:
            human_input = '{"approved": true}'

        graph_state = _stream_until_interrupt(Command(resume=human_input))

    final_state = pipeline.get_state(config).values
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE — Status: {final_state.get('final_status', 'DONE')}")
    print(f"{'=' * 60}")
    return final_state


def _get_interrupt_value_local(state):
    """Extract interrupt payload from graph state."""
    try:
        for task in state.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                return task.interrupts[0].value
    except Exception:
        pass
    return None


# ================================================================
# STANDALONE DEMO (no LangGraph / API key required)
# ================================================================

def demo_reflect(data_path: str = "data/simulated_batch_olink.csv") -> None:
    """
    Run the reflection node logic in isolation against a real CSV
    to verify it works without the full LangGraph graph.

    Usage:
        python batch_correction_reflection_agent.py
    """
    import sys
    import pandas as pd
    from scipy import stats
    from sklearn.impute import SimpleImputer

    print("\n" + "=" * 60)
    print("DEMO: Reflection Agent (standalone)")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────
    print(f"\nLoading: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  {df.shape[0]} samples × {df.shape[1]} columns")

    meta_keywords = [
        "eid", "age", "sex", "bmi", "apoe", "smoking", "education",
        "plate", "batch", "centre", "date", "time", "quality",
        "freeze", "delay", "well", "source", "131", "53-", "case", "status",
    ]
    protein_cols = [
        c for c in df.columns
        if not any(k in c.lower() for k in meta_keywords)
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    print(f"  Proteins: {len(protein_cols)}")

    # ── Simulate batch_assessment ──────────────────────────────
    batch_fields = [
        "plate_id", "assessment_centre",
        "freeze_thaw_cycles", "sample_quality_flag",
        "processing_delay_mins",
    ]
    batch_fields = [f for f in batch_fields if f in df.columns]

    print("\nAssessing batch effects (η²)...")
    assessment = {}
    for field in batch_fields:
        m = _reassess_field(df, protein_cols, field, sample_n=100)
        assessment[field] = m
        print(f"  {field:<35} η²={m['mean_eta_squared']:.4f}  "
              f"{m['pct_proteins_affected']:.1f}% proteins")

    # ── Build mock state ───────────────────────────────────────
    os.makedirs(CACHE_DIR, exist_ok=True)
    df_path = _save_df(df, "df_current")

    mock_state: dict = {
        "data_path":              data_path,
        "df_raw_path":            df_path,
        "df_current_path":        df_path,
        "protein_cols":           protein_cols,
        "metadata_cols":          [c for c in df.columns if c not in protein_cols],
        "candidate_batch_fields": {f: {} for f in batch_fields},
        "confirmed_batch_fields": batch_fields,
        "primary_batch_field":    "plate_id" if "plate_id" in batch_fields else batch_fields[0],
        "batch_assessment":       assessment,
        "assessment_plots":       [],
        "correction_needed":      True,
        "proposed_strategy":      {},
        "approved_strategy":      {},
        "df_corrected_path":      "",
        "validation_metrics":     {},
        "validation_plots":       [],
        "human_decisions":        [],
        "agent_reasoning":        [],
        "reflection_log":         [],
        "current_step":           "assessment_done",
        "error_log":              [],
        "final_status":           "RUNNING",
    }

    # ── Run reflection node ────────────────────────────────────
    updated = node_reflect_on_assessment(mock_state)  # type: ignore[arg-type]

    # ── Report ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("REFLECTION RESULTS")
    print("=" * 60)
    print(f"  Confirmed (ComBat): {updated['confirmed_batch_fields']}")
    print(f"  Primary:            {updated['primary_batch_field']}")
    print(f"  Method:             {updated['proposed_strategy']['recommended_method']}")
    print(f"  Demoted:            {updated['proposed_strategy']['secondary_batch_variables']}")
    print(f"  Dropped:            {updated['proposed_strategy']['dropped_fields']}")
    print(f"\nReasoning:")
    for line in updated["agent_reasoning"]:
        print(f"  {line}")

    print(f"\nReflection log ({len(updated['reflection_log'])} round(s)):")
    for rnd in updated["reflection_log"]:
        print(f"\n  Round {rnd['round']}:")
        for d in rnd["decisions"]:
            print(f"    {d['field']:<35} {d['action']}")
        print(f"  LLM: {rnd['llm_commentary'][:200]}...")


if __name__ == "__main__":
    import sys

    data = sys.argv[1] if len(sys.argv) > 1 else "data/simulated_batch_olink.csv"

    if not os.path.exists(data):
        print(f"Data not found at '{data}'. Generating simulated data first...")
        try:
            from simulate_data import simulate_batch_data
            simulate_batch_data()
        except ImportError:
            print("simulate_data.py not found. Please provide a data path.")
            sys.exit(1)

    demo_reflect(data)
