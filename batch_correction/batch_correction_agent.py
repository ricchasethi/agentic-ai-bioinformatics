"""
================================================================
Human-in-the-Loop Agentic Batch Correction Pipeline
UK Biobank OLINK Proteomics Data
================================================================

Architecture:
  LangGraph StateGraph with 8 nodes + 3 human approval checkpoints

  [load_data] → [detect_batch_fields] → [HUMAN: confirm fields]
      → [assess_batch_effects] → [HUMAN: approve correction strategy]
      → [apply_correction] → [validate_correction]
      → [HUMAN: accept/reject results] → [save_outputs]

Project conventions, constraints, and data field definitions are
documented in CLAUDE.md at the project root. Read it before editing.

Key rules enforced here:
  - State contains NO DataFrames — use _save_df/_load_df with parquet cache
  - All dicts stored in state must pass through _sanitise() first
  - ComBat covar_mod must always include AD_case, age, sex
  - QC order: filter → winsorise → impute → batch correct → INT → analyse

Dependencies:
    pip install langgraph langchain langchain-anthropic
    pip install pandas numpy scipy scikit-learn matplotlib seaborn
    pip install inmoose pyarrow   # ComBat + parquet cache
"""

from __future__ import annotations

import json
import os
from typing import Literal, TypedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# LangGraph imports
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

# LangChain / Anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage


# ================================================================
# STATE DEFINITION
# ================================================================

class BatchCorrectionState(TypedDict):
    """
    Shared state passed between all nodes in the graph.
    Every node reads from and writes to this state object.

    IMPORTANT: LangGraph's MemorySaver serialises state to msgpack after
    every node. pandas DataFrames are NOT msgpack-serialisable, so we
    never store them directly in state. Instead each node saves its
    DataFrame to a parquet file under results/batch/cache/ and stores
    only the file path as a plain string.
    """
    # --- Data paths (never store DataFrames directly) ---
    data_path: str
    df_raw_path: str        # path to raw DataFrame parquet
    df_current_path: str    # path to working DataFrame parquet
    protein_cols: list[str]
    metadata_cols: list[str]

    # --- Batch field detection ---
    candidate_batch_fields: dict       # {field_name: {type, n_unique, description}}
    confirmed_batch_fields: list[str]  # human-confirmed fields to use
    primary_batch_field: str           # the main batch variable for ComBat

    # --- Batch assessment results ---
    batch_assessment: dict             # {field: {pct_affected, mean_eta_sq, ...}}
    assessment_plots: list[str]        # paths to generated plots
    correction_needed: bool

    # --- Correction strategy ---
    proposed_strategy: dict            # AI-proposed strategy with reasoning
    approved_strategy: dict            # human-approved (may be modified)

    # --- Correction results ---
    df_corrected_path: str             # path to corrected DataFrame parquet
    validation_metrics: dict           # before/after comparison
    validation_plots: list[str]

    # --- Human interaction ---
    human_decisions: list[dict]        # audit trail of all human decisions
    agent_reasoning: list[str]         # AI reasoning log

    # --- Control flow ---
    current_step: str
    error_log: list[str]
    final_status: str


# ================================================================
# DATAFRAME CACHE HELPERS
# ================================================================

CACHE_DIR = "results/batch/cache"

def _save_df(df: pd.DataFrame, name: str) -> str:
    """Save a DataFrame to parquet and return the path."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{name}.parquet")
    df.to_parquet(path, index=True)
    return path

def _load_df(path: str) -> pd.DataFrame:
    """Load a DataFrame from a parquet path stored in state."""
    if not path:
        raise ValueError("DataFrame path is empty — node dependency not met.")
    return pd.read_parquet(path)


def _sanitise(obj):
    """
    Recursively convert numpy scalars and bools to plain Python types.

    MUST be called on every dict/list before storing into LangGraph state.
    numpy.bool_, numpy.int64, numpy.float64 all fail msgpack serialisation.
    See CLAUDE.md §"Never store DataFrames in LangGraph state" for full rules.

    Common traps:
      df[col].nunique() < 50   → np.bool_  → wrap in bool()
      np.mean(arr) > threshold → np.bool_  → wrap in bool()
      any(...)                 → safe, always Python bool
      sum(...)                 → safe, always Python int
    """
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ================================================================
# LLM SETUP
# ================================================================

def get_llm(model: str = "claude-sonnet-4-20250514") -> ChatAnthropic:
    """Initialise the Anthropic LLM."""
    return ChatAnthropic(
        model=model,
        max_tokens=2048,
        temperature=0.1,   # low temp for analytical consistency
    )


SYSTEM_PROMPT = """You are an expert bioinformatician specialising in UK Biobank 
proteomics data analysis and batch effect correction. You have deep knowledge of:
- OLINK NPX technology and its technical artefacts
- ComBat and other batch correction methods
- UK Biobank data fields and their biological/technical meanings
- Statistical methods for batch effect quantification (ANOVA, eta-squared)

When analysing data, be precise, quantitative, and conservative. Always explain 
your reasoning. Flag any concerns clearly. Output structured JSON when requested."""


# ================================================================
# NODE 1: LOAD DATA
# ================================================================

def node_load_data(state: BatchCorrectionState) -> BatchCorrectionState:
    """
    Load OLINK data and perform initial inventory.
    Identifies protein columns vs metadata columns automatically.
    Saves DataFrames to parquet cache — never puts them in state directly.
    """
    print("\n[NODE 1] Loading data...")

    try:
        df = pd.read_csv(state["data_path"])
    except Exception as e:
        state["error_log"].append(f"Data load failed: {e}")
        state["final_status"] = "ERROR"
        return state

    # Heuristic: protein columns are float columns that are NOT known metadata
    known_meta_patterns = [
        'eid', 'age', 'sex', 'bmi', 'smoking', 'education', 'apoe',
        'AD_case', 'AD_status', 'days_to', 'PC', '131', '53-',
        'plate', 'batch', 'centre', 'date', 'time', 'quality',
        'freeze', 'processing', 'well', 'source', 'collection'
    ]

    protein_cols, metadata_cols = [], []
    for col in df.columns:
        col_lower = col.lower()
        is_meta = any(pat in col_lower for pat in known_meta_patterns)
        if is_meta or df[col].dtype == object:
            metadata_cols.append(col)
        else:
            protein_cols.append(col)

    print(f"  Loaded: {df.shape[0]} samples × {df.shape[1]} columns")
    print(f"  Protein columns detected: {len(protein_cols)}")
    print(f"  Metadata columns detected: {len(metadata_cols)}")

    # Save DataFrames to disk — store only paths in state
    raw_path     = _save_df(df, "df_raw")
    current_path = _save_df(df.copy(), "df_current")

    state["df_raw_path"]     = raw_path
    state["df_current_path"] = current_path
    state["protein_cols"]    = protein_cols
    state["metadata_cols"]   = metadata_cols
    state["current_step"]    = "data_loaded"
    state["agent_reasoning"].append(
        f"Loaded {df.shape[0]} samples with {len(protein_cols)} protein columns "
        f"and {len(metadata_cols)} metadata columns."
    )

    return state


# ================================================================
# NODE 2: DETECT BATCH FIELDS (AI Agent)
# ================================================================

def node_detect_batch_fields(state: BatchCorrectionState) -> BatchCorrectionState:
    """
    AI agent scans metadata columns and identifies which ones
    are likely batch-relevant using LLM reasoning + statistical checks.
    """
    print("\n[NODE 2] AI agent detecting batch-relevant fields...")

    df = _load_df(state["df_current_path"])
    meta_cols = state["metadata_cols"]

    # Build a statistical summary of each metadata column
    col_summaries = {}
    for col in meta_cols:
        if col not in df.columns:
            continue
        series = df[col]
        summary = {
            "dtype":    str(series.dtype),
            "n_unique": int(series.nunique()),
            "pct_null": float(series.isna().mean()),
            "sample_values": series.dropna().astype(str).head(5).tolist()
        }
        if pd.api.types.is_numeric_dtype(series):
            summary["min"]  = float(series.min())
            summary["max"]  = float(series.max())
            summary["mean"] = float(series.mean())
        col_summaries[col] = summary

    # Ask LLM to classify each column
    llm = get_llm()

    prompt = f"""
You are analysing UK Biobank OLINK proteomics metadata columns to identify 
which are batch-relevant (i.e., could introduce technical variation in protein measurements).

Here are the metadata columns and their statistics:
{json.dumps(col_summaries, indent=2)}

UK Biobank OLINK batch-relevant field categories:
1. PLATE/BATCH: plate_id, batch_id (primary ComBat variable)
2. SPATIAL: well_position, plate_row, plate_column (within-plate effects)  
3. TEMPORAL: collection_date, processing_date (reagent lot, freeze-thaw)
4. CENTRE: assessment_centre, recruitment_site (pre-analytical handling)
5. SAMPLE_QUALITY: haemolysis_flag, freeze_thaw_count, processing_delay
6. TIME_OF_DAY: venepuncture_time (circadian protein variation)

For EACH column, classify it as:
- "primary_batch": best single variable for ComBat (usually plate_id)
- "secondary_batch": additional batch covariate
- "biological": biological confounder (NOT batch — do not correct)
- "outcome": case/control status — CRITICAL: never batch-correct on this
- "irrelevant": not relevant to batch correction

Return ONLY valid JSON in this exact format:
{{
  "classifications": {{
    "column_name": {{
      "category": "primary_batch|secondary_batch|biological|outcome|irrelevant",
      "confidence": 0.0-1.0,
      "reasoning": "brief explanation"
    }}
  }},
  "recommended_primary_batch": "column_name or null",
  "summary": "2-3 sentence summary of batch structure found"
}}
"""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    try:
        response = llm.invoke(messages)
        # Parse JSON from response
        raw = response.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        classifications = json.loads(raw.strip())
    except Exception as e:
        # Fallback: heuristic classification
        print(f"  LLM classification failed ({e}), using heuristics")
        classifications = _heuristic_batch_classification(col_summaries)

    # Extract candidate batch fields
    candidate_fields = {}
    for col, info in classifications.get("classifications", {}).items():
        if info["category"] in ("primary_batch", "secondary_batch"):
            candidate_fields[col] = {
                "category":   info["category"],
                "confidence": info["confidence"],
                "reasoning":  info["reasoning"],
                "n_unique":   col_summaries.get(col, {}).get("n_unique", "?"),
                "pct_null":   col_summaries.get(col, {}).get("pct_null", "?"),
            }

    state["candidate_batch_fields"] = _sanitise(candidate_fields)
    state["agent_reasoning"].append(
        classifications.get("summary", "Batch field detection complete.")
    )

    print(f"  Candidate batch fields identified: {list(candidate_fields.keys())}")
    print(f"  Recommended primary: {classifications.get('recommended_primary_batch')}")

    return state


def _heuristic_batch_classification(col_summaries: dict) -> dict:
    """Fallback heuristic when LLM is unavailable."""
    PLATE_KEYWORDS = ['plate', 'batch', 'lot']
    SPATIAL_KEYWORDS = ['well', 'row', 'column', 'position']
    TIME_KEYWORDS = ['date', 'time', 'day']
    CENTRE_KEYWORDS = ['centre', 'center', 'site', 'location']
    QUALITY_KEYWORDS = ['quality', 'haemo', 'freeze', 'delay', 'flag']

    classifications = {}
    for col, stats_dict in col_summaries.items():
        col_lower = col.lower()
        if any(k in col_lower for k in PLATE_KEYWORDS):
            cat = "primary_batch"
        elif any(k in col_lower for k in SPATIAL_KEYWORDS + TIME_KEYWORDS + CENTRE_KEYWORDS + QUALITY_KEYWORDS):
            cat = "secondary_batch"
        elif col_lower in ('ad_case', 'outcome', 'disease'):
            cat = "outcome"
        elif col_lower in ('age', 'sex', 'bmi', 'apoe_e4'):
            cat = "biological"
        else:
            cat = "irrelevant"

        classifications[col] = {
            "category": cat, "confidence": 0.7,
            "reasoning": "Heuristic classification"
        }

    primary = next((c for c, v in classifications.items()
                     if v["category"] == "primary_batch"), None)

    return {"classifications": classifications,
            "recommended_primary_batch": primary,
            "summary": "Heuristic batch field classification applied."}


# ================================================================
# HUMAN CHECKPOINT 1: Confirm Batch Fields
# ================================================================

def human_confirm_batch_fields(state: BatchCorrectionState) -> BatchCorrectionState:
    """
    Human-in-the-loop checkpoint 1.
    Shows AI-detected batch fields and asks human to confirm/modify.
    Uses LangGraph interrupt() for async suspension.
    """
    print("\n" + "="*60)
    print("HUMAN CHECKPOINT 1: Confirm Batch Fields")
    print("="*60)

    candidates = state["candidate_batch_fields"]

    # Format the decision request for the human
    display = "\nAI-detected batch-relevant fields:\n"
    for i, (col, info) in enumerate(candidates.items()):
        display += (f"\n  [{i+1}] {col}\n"
                    f"       Category:   {info['category']}\n"
                    f"       Confidence: {info['confidence']:.0%}\n"
                    f"       N unique:   {info['n_unique']}\n"
                    f"       Reasoning:  {info['reasoning']}\n")

    display += "\nAI Reasoning:\n" + "\n".join(state["agent_reasoning"][-3:])

    # Suspend execution and wait for human input via interrupt()
    human_response = interrupt({
        "checkpoint": "confirm_batch_fields",
        "message": display,
        "candidate_fields": candidates,
        "instructions": (
            "Please review the detected batch fields. "
            "Respond with a JSON object: "
            "{'confirmed_fields': ['col1', 'col2'], "
            "'primary_batch_field': 'col1', "
            "'notes': 'any comments'}"
        )
    })

    # Process human response
    if isinstance(human_response, str):
        try:
            response_data = json.loads(human_response)
        except json.JSONDecodeError:
            # If human just typed field names as comma-separated
            confirmed = [f.strip() for f in human_response.split(",")]
            response_data = {
                "confirmed_fields": confirmed,
                "primary_batch_field": confirmed[0] if confirmed else "",
                "notes": ""
            }
    else:
        response_data = human_response

    confirmed   = response_data.get("confirmed_fields", list(candidates.keys()))
    primary     = response_data.get("primary_batch_field",
                                     confirmed[0] if confirmed else "")

    state["confirmed_batch_fields"] = confirmed
    state["primary_batch_field"]    = primary
    state["human_decisions"].append({
        "checkpoint": "confirm_batch_fields",
        "input":      display,
        "response":   response_data
    })

    print(f"\n  Human confirmed fields: {confirmed}")
    print(f"  Primary batch field:    {primary}")

    return state


# ================================================================
# NODE 3: ASSESS BATCH EFFECTS
# ================================================================

def node_assess_batch_effects(state: BatchCorrectionState) -> BatchCorrectionState:
    """
    Quantify batch effects for each confirmed batch field.
    Computes η² (eta-squared) — variance explained by batch.
    Generates PCA plots coloured by batch.
    """
    print("\n[NODE 3] Assessing batch effects...")

    df           = _load_df(state["df_current_path"])
    protein_cols = state["protein_cols"]
    batch_fields = state["confirmed_batch_fields"]

    os.makedirs("results/batch", exist_ok=True)
    assessment  = {}
    plot_paths  = []

    for batch_field in batch_fields:
        if batch_field not in df.columns:
            continue

        batch_var    = df[batch_field]
        is_categorical = bool(df[batch_field].dtype == object or
                               df[batch_field].nunique() < 50)

        eta_sq_vals, p_vals = [], []

        for prot in protein_cols:
            y = df[prot].dropna()
            x = batch_var.loc[y.index].dropna()
            y = y.loc[x.index]

            if len(y) < 20:
                continue

            if is_categorical:
                groups = [y[x == cat].values for cat in x.unique()
                           if len(y[x == cat]) > 2]
                if len(groups) < 2:
                    continue
                f_stat, p_val = stats.f_oneway(*groups)
                grand_mean = y.mean()
                ss_between = sum(len(g)*(g.mean()-grand_mean)**2 for g in groups)
                ss_total   = ((y - grand_mean)**2).sum()
                eta_sq     = float(ss_between / ss_total) if ss_total > 0 else 0
            else:
                rho, p_val = stats.spearmanr(x, y)
                eta_sq     = float(rho**2)

            eta_sq_vals.append(eta_sq)
            p_vals.append(float(p_val))

        n_sig        = int(sum(p < 0.05 for p in p_vals))
        pct_affected = float(n_sig / len(protein_cols) * 100)
        mean_eta_sq  = float(np.mean(eta_sq_vals)) if eta_sq_vals else 0.0

        # Severity classification
        if pct_affected > 30 or mean_eta_sq > 0.05:
            severity = "HIGH — ComBat correction required"
        elif pct_affected > 10 or mean_eta_sq > 0.01:
            severity = "MODERATE — ComBat recommended"
        else:
            severity = "LOW — covariate adjustment sufficient"

        assessment[batch_field] = {
            "pct_proteins_affected": pct_affected,
            "mean_eta_squared":      mean_eta_sq,
            "n_proteins_significant":n_sig,
            "severity":              severity,
            "is_categorical":        is_categorical
        }

        print(f"  {batch_field}: {pct_affected:.1f}% proteins affected | "
              f"η²={mean_eta_sq:.4f} | {severity}")

        # PCA plot coloured by this batch field
        plot_path = _plot_batch_pca(df, protein_cols, batch_field,
                                     f"results/batch/pca_{batch_field}.png")
        if plot_path:
            plot_paths.append(plot_path)

    # Overall correction decision
    correction_needed = any(
        a["pct_proteins_affected"] > 10
        for a in assessment.values()
    )

    state["batch_assessment"]   = _sanitise(assessment)
    state["assessment_plots"]   = plot_paths
    state["correction_needed"]  = bool(correction_needed)
    state["agent_reasoning"].append(
        f"Batch assessment complete. Correction needed: {correction_needed}. "
        f"Fields assessed: {list(assessment.keys())}"
    )

    # Update df_current_path (df unchanged in this node, but re-save to be explicit)
    state["df_current_path"] = _save_df(df, "df_current")

    return state


def _plot_batch_pca(df, protein_cols, colour_col, save_path):
    """Generate PCA plot coloured by a batch variable."""
    try:
        X = SimpleImputer(strategy='median').fit_transform(
            df[protein_cols].select_dtypes(include=[np.number])
        )
        X = StandardScaler().fit_transform(X)
        pcs = PCA(n_components=2, random_state=42).fit_transform(X)

        fig, ax = plt.subplots(figsize=(8, 6))
        c_vals = df[colour_col].values

        if df[colour_col].nunique() <= 20:
            unique_vals = df[colour_col].dropna().unique()
            cmap = plt.cm.get_cmap('tab20', len(unique_vals))
            for i, val in enumerate(unique_vals):
                mask = c_vals == val
                ax.scatter(pcs[mask, 0], pcs[mask, 1],
                            color=cmap(i), label=str(val),
                            alpha=0.5, s=12)
            if len(unique_vals) <= 10:
                ax.legend(title=colour_col, fontsize=7, markerscale=2)
        else:
            sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=pd.to_numeric(c_vals, errors='coerce'),
                             cmap='RdYlBu_r', alpha=0.5, s=12)
            plt.colorbar(sc, ax=ax, label=colour_col)

        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title(f"PCA coloured by: {colour_col}")
        plt.tight_layout()
        plt.savefig(save_path, dpi=130, bbox_inches='tight')
        plt.close()
        return save_path
    except Exception as e:
        print(f"  PCA plot failed for {colour_col}: {e}")
        return None


# ================================================================
# NODE 4: PROPOSE CORRECTION STRATEGY (AI Agent)
# ================================================================

def node_propose_strategy(state: BatchCorrectionState) -> BatchCorrectionState:
    """
    AI agent analyses batch assessment results and proposes
    a detailed correction strategy with full justification.
    """
    print("\n[NODE 4] AI agent proposing correction strategy...")

    assessment = state["batch_assessment"]
    primary    = state["primary_batch_field"]

    llm = get_llm()

    prompt = f"""
You are advising on batch correction strategy for UK Biobank OLINK proteomics data.

BATCH ASSESSMENT RESULTS:
{json.dumps(assessment, indent=2)}

PRIMARY BATCH FIELD: {primary}
CORRECTION NEEDED: {state['correction_needed']}

Available correction methods:
1. ComBat (pyComBat/sva) — Empirical Bayes, gold standard for proteomics
   - Best for: plate/batch effects with ≥10 samples per batch
   - Risk: over-correction if biological signal correlates with batch
   
2. Covariate adjustment — Include batch as covariate in regression
   - Best for: low-severity effects (η² < 0.01)
   - Risk: does not remove batch from data itself

3. Mixed model (limma::removeBatchEffect equivalent) 
   - Best for: complex nested batch structures
   - Risk: computationally intensive

4. No correction — proceed with batch as covariate only
   - Best for: η² < 0.005, <5% proteins affected

CRITICAL RULES:
- NEVER include AD_case/outcome as a batch variable in ComBat
- ALWAYS protect biological covariates (age, sex, AD_case) in ComBat mod matrix
- If primary batch field has batches with <10 samples, recommend merging first
- For multiple batch variables, correct sequentially (primary first)

Based on the assessment, propose a correction strategy.
Return ONLY valid JSON:
{{
  "recommended_method": "ComBat|covariate_only|mixed_model|none",
  "primary_batch_variable": "column_name",
  "secondary_batch_variables": ["col1"],
  "biological_covariates_to_protect": ["age", "sex", "AD_case"],
  "pre_correction_steps": ["step1", "step2"],
  "combat_parameters": {{
    "parametric": true,
    "mean_only": false,
    "ref_batch": null
  }},
  "justification": "detailed explanation",
  "risks_and_mitigations": ["risk1 → mitigation1"],
  "validation_checks": ["check1", "check2"]
}}
"""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        strategy = json.loads(raw.strip())
    except Exception as e:
        print(f"  LLM strategy proposal failed ({e}), using defaults")
        strategy = _default_strategy(assessment, primary)

    state["proposed_strategy"] = _sanitise(strategy)
    state["agent_reasoning"].append(
        f"Strategy proposed: {strategy.get('recommended_method')}. "
        f"Justification: {strategy.get('justification', '')[:200]}"
    )

    print(f"  Proposed method: {strategy.get('recommended_method')}")
    print(f"  Justification: {strategy.get('justification', '')[:150]}...")

    return state


def _default_strategy(assessment, primary):
    """Default strategy when LLM unavailable."""
    max_pct = max((a["pct_proteins_affected"] for a in assessment.values()), default=0)
    method  = "ComBat" if max_pct > 10 else "covariate_only"
    return {
        "recommended_method": method,
        "primary_batch_variable": primary,
        "secondary_batch_variables": [],
        "biological_covariates_to_protect": ["age", "sex", "AD_case"],
        "pre_correction_steps": ["Check minimum batch sizes", "Merge small batches"],
        "combat_parameters": {"parametric": True, "mean_only": False, "ref_batch": None},
        "justification": f"Default strategy based on {max_pct:.1f}% proteins affected.",
        "risks_and_mitigations": ["Over-correction → protect biological covariates in mod"],
        "validation_checks": ["PCA before/after", "Known AD marker preservation"]
    }


# ================================================================
# HUMAN CHECKPOINT 2: Approve Correction Strategy
# ================================================================

def human_approve_strategy(state: BatchCorrectionState) -> BatchCorrectionState:
    """
    Human-in-the-loop checkpoint 2.
    Human reviews and approves/modifies the proposed correction strategy.
    """
    print("\n" + "="*60)
    print("HUMAN CHECKPOINT 2: Approve Correction Strategy")
    print("="*60)

    strategy   = state["proposed_strategy"]
    assessment = state["batch_assessment"]

    display = f"""
BATCH EFFECT SUMMARY:
{"─"*40}"""
    for field, metrics in assessment.items():
        display += (f"\n  {field}: {metrics['pct_proteins_affected']:.1f}% proteins affected "
                    f"| η²={metrics['mean_eta_squared']:.4f} | {metrics['severity']}")

    display += f"""

AI-PROPOSED CORRECTION STRATEGY:
{"─"*40}
  Method:            {strategy.get('recommended_method')}
  Primary batch var: {strategy.get('primary_batch_variable')}
  Secondary vars:    {strategy.get('secondary_batch_variables')}
  Protected vars:    {strategy.get('biological_covariates_to_protect')}
  
  Justification:
  {strategy.get('justification', '')}
  
  Risks & Mitigations:
  {chr(10).join('  - ' + r for r in strategy.get('risks_and_mitigations', []))}
  
  Validation Checks:
  {chr(10).join('  - ' + c for c in strategy.get('validation_checks', []))}

Assessment plots saved to: results/batch/
"""

    human_response = interrupt({
        "checkpoint": "approve_strategy",
        "message":    display,
        "proposed_strategy": strategy,
        "instructions": (
            "Review the proposed strategy. You can: "
            "(1) Approve as-is: respond with {'approved': true} "
            "(2) Modify: respond with {'approved': true, 'modifications': {...}} "
            "(3) Reject: respond with {'approved': false, 'reason': '...'}"
        )
    })

    if isinstance(human_response, str):
        try:
            response_data = json.loads(human_response)
        except Exception:
            response_data = {"approved": True}
    else:
        response_data = human_response

    if response_data.get("approved", True):
        approved_strategy = strategy.copy()
        if "modifications" in response_data:
            approved_strategy.update(response_data["modifications"])
        state["approved_strategy"] = approved_strategy
        print(f"  Strategy approved: {approved_strategy.get('recommended_method')}")
    else:
        # Human rejected — fall back to covariate only
        state["approved_strategy"] = {
            **strategy,
            "recommended_method": "covariate_only",
            "rejection_reason": response_data.get("reason", "Human rejected")
        }
        print(f"  Strategy rejected. Using covariate_only as fallback.")

    state["human_decisions"].append({
        "checkpoint": "approve_strategy",
        "input":      display,
        "response":   response_data
    })

    return state


# ================================================================
# NODE 5: APPLY BATCH CORRECTION
# ================================================================

def node_apply_correction(state: BatchCorrectionState) -> BatchCorrectionState:
    """
    Apply the approved batch correction strategy.
    Supports ComBat (via pyComBat/inmoose) and covariate-only approaches.
    """
    print("\n[NODE 5] Applying batch correction...")

    strategy     = state["approved_strategy"]
    df           = _load_df(state["df_current_path"])
    protein_cols = state["protein_cols"]
    method       = strategy.get("recommended_method", "covariate_only")
    batch_col    = strategy.get("primary_batch_variable", "")
    protect_vars = strategy.get("biological_covariates_to_protect", [])

    if method == "ComBat":
        df_corrected = _apply_combat(df, protein_cols, batch_col, protect_vars)
    elif method == "covariate_only":
        df_corrected = df.copy()
        print(f"  Covariate-only: batch field '{batch_col}' retained for use in regression.")
        print("  No data transformation applied.")
    else:
        print(f"  Unknown method '{method}', defaulting to covariate_only.")
        df_corrected = df.copy()

    state["df_corrected_path"] = _save_df(df_corrected, "df_corrected")
    state["agent_reasoning"].append(
        f"Batch correction applied: method={method}, batch_col={batch_col}"
    )

    return state


def _apply_combat(df, protein_cols, batch_col, protect_vars):
    """
    Apply ComBat correction using pyComBat (inmoose package).

    Convention (see CLAUDE.md §ComBat):
      - pyComBat expects proteins × samples — always pass matrix.T
      - covar_mod must include AD_case, age, sex to protect biological signal
      - Batches with <3 samples are dropped before correction
      - The "[WARNING] covariate matrix transposed" message is expected and safe
    """
    if batch_col not in df.columns:
        print(f"  ERROR: batch column '{batch_col}' not found. Skipping correction.")
        return df

    # Check minimum batch sizes
    batch_counts = df[batch_col].value_counts()
    small_batches = batch_counts[batch_counts < 3].index.tolist()
    if small_batches:
        print(f"  WARNING: {len(small_batches)} batches with <3 samples — removing them")
        df = df[~df[batch_col].isin(small_batches)]

    # Impute proteins before ComBat (ComBat requires complete matrix)
    protein_matrix = df[protein_cols].copy()
    imputer = SimpleImputer(strategy='median')
    protein_matrix_imputed = pd.DataFrame(
        imputer.fit_transform(protein_matrix),
        columns=protein_cols, index=df.index
    )

    batch_labels = df[batch_col].astype(str).values
    covar_df = None
    valid_protect = [v for v in protect_vars if v in df.columns]
    if valid_protect:
        covar_df = df[valid_protect].fillna(df[valid_protect].median())

    try:
        from inmoose.pycombat import pycombat_norm
        print(f"  Running pyComBat on {len(protein_cols)} proteins × "
              f"{len(df)} samples ({len(batch_counts)} batches)...")

        corrected = pycombat_norm(
            protein_matrix_imputed.T,
            batch_labels,
            covar_mod=covar_df.T if covar_df is not None else None
        )
        df_corrected = df.copy()
        df_corrected[protein_cols] = corrected.T.values
        print("  ComBat correction complete.")
        return df_corrected

    except ImportError:
        print("  inmoose not installed — simulating correction for demonstration.")
        return _simulate_combat_correction(df, protein_cols, batch_col)


def _simulate_combat_correction(df, protein_cols, batch_col):
    """
    Simple mean-centering per batch as a ComBat stand-in for testing.
    In production, always use the real ComBat implementation.
    """
    df_corrected = df.copy()
    grand_means  = df[protein_cols].mean()

    for batch_val in df[batch_col].unique():
        mask = df[batch_col] == batch_val
        batch_means = df.loc[mask, protein_cols].mean()
        shift = grand_means - batch_means
        df_corrected.loc[mask, protein_cols] = \
            df.loc[mask, protein_cols] + shift

    return df_corrected


# ================================================================
# NODE 6: VALIDATE CORRECTION
# ================================================================

def node_validate_correction(state: BatchCorrectionState) -> BatchCorrectionState:
    """
    Validate that batch correction:
    1. Reduced batch-associated variance (η² decreased)
    2. Preserved biological signal (known AD markers still significant)
    3. Did not introduce new artefacts (protein distributions look normal)
    """
    print("\n[NODE 6] Validating batch correction...")

    df_raw       = _load_df(state["df_current_path"])
    df_corrected = _load_df(state["df_corrected_path"])
    protein_cols = state["protein_cols"]
    batch_col    = state["primary_batch_field"]
    known_markers= ['NEFL', 'GFAP', 'TREM2', 'CLU', 'APOE', 'APP']

    os.makedirs("results/batch", exist_ok=True)
    metrics = {}
    plot_paths = []

    # --- 1. Batch variance reduction ---
    if batch_col in df_raw.columns:
        eta_before, eta_after = [], []
        sample_proteins = [p for p in protein_cols[:100] if p in df_raw.columns]

        for prot in sample_proteins:
            for df_, store in [(df_raw, eta_before), (df_corrected, eta_after)]:
                y = df_[prot].dropna()
                x = df_.loc[y.index, batch_col].dropna()
                y = y.loc[x.index]
                groups = [y[x == c].values for c in x.unique() if len(y[x == c]) > 1]
                if len(groups) < 2:
                    store.append(0.0)
                    continue
                grand = y.mean()
                ss_b  = sum(len(g)*(g.mean()-grand)**2 for g in groups)
                ss_t  = ((y-grand)**2).sum()
                store.append(float(ss_b/ss_t) if ss_t > 0 else 0.0)

        reduction = (1 - np.mean(eta_after) / max(np.mean(eta_before), 1e-9)) * 100
        metrics["batch_variance_reduction"] = {
            "eta_sq_before": float(np.mean(eta_before)),
            "eta_sq_after":  float(np.mean(eta_after)),
            "pct_reduction": float(reduction),
            "adequate":      bool(reduction > 50)
        }
        print(f"  Batch variance: η²={np.mean(eta_before):.4f} → "
              f"{np.mean(eta_after):.4f} ({reduction:.1f}% reduction)")

    # --- 2. Biological signal preservation ---
    if 'AD_case' in df_raw.columns:
        signal_before, signal_after = {}, {}
        for marker in known_markers:
            if marker not in df_raw.columns:
                continue
            for df_, store in [(df_raw, signal_before), (df_corrected, signal_after)]:
                cases    = df_.loc[df_['AD_case'] == 1, marker].dropna()
                controls = df_.loc[df_['AD_case'] == 0, marker].dropna()
                if len(cases) < 5 or len(controls) < 5:
                    continue
                _, pval  = stats.ttest_ind(cases, controls)
                effect   = (cases.mean() - controls.mean()) / df_[marker].std()
                store[marker] = {"pvalue": float(pval), "effect_size": float(effect)}

        metrics["biological_signal_preservation"] = {
            "markers_checked": list(signal_before.keys()),
            "signal_before":   signal_before,
            "signal_after":    signal_after,
        }

        for m in signal_before:
            if m in signal_after:
                d_eff = signal_after[m]["effect_size"] - signal_before[m]["effect_size"]
                print(f"  {m}: effect size change = {d_eff:+.3f} "
                      f"({'OK' if abs(d_eff) < 0.1 else 'WARNING: large change'})")

    # --- 3. Distribution comparison plot ---
    plot_path = _plot_before_after(df_raw, df_corrected, protein_cols, batch_col)
    if plot_path:
        plot_paths.append(plot_path)

    state["validation_metrics"] = _sanitise(metrics)
    state["validation_plots"]   = plot_paths
    state["agent_reasoning"].append(
        f"Validation complete. "
        f"Batch variance reduction: {metrics.get('batch_variance_reduction', {}).get('pct_reduction', 0):.1f}%"
    )

    return state


def _plot_before_after(df_raw, df_corrected, protein_cols, batch_col):
    """Side-by-side PCA before vs after correction."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        valid_proteins = [p for p in protein_cols if p in df_raw.columns][:200]

        for ax, df_, title in [
            (axes[0], df_raw,       "Before Correction"),
            (axes[1], df_corrected, "After Correction")
        ]:
            X = SimpleImputer(strategy='median').fit_transform(df_[valid_proteins])
            X = StandardScaler().fit_transform(X)
            pcs = PCA(n_components=2, random_state=42).fit_transform(X)

            if batch_col and batch_col in df_.columns:
                unique_batches = df_[batch_col].unique()
                cmap = plt.cm.get_cmap('tab20', min(len(unique_batches), 20))
                for i, bv in enumerate(unique_batches[:20]):
                    mask = (df_[batch_col] == bv).values
                    ax.scatter(pcs[mask, 0], pcs[mask, 1],
                                color=cmap(i % 20), alpha=0.4, s=10,
                                label=str(bv) if i < 8 else "")
            else:
                ax.scatter(pcs[:, 0], pcs[:, 1], alpha=0.4, s=10, color='steelblue')

            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

        plt.suptitle("Batch Effect: Before vs After Correction", fontsize=14, y=1.01)
        plt.tight_layout()
        path = "results/batch/before_after_pca.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path
    except Exception as e:
        print(f"  Before/after plot failed: {e}")
        return None


# ================================================================
# HUMAN CHECKPOINT 3: Accept/Reject Results
# ================================================================

def human_accept_results(state: BatchCorrectionState) -> BatchCorrectionState:
    """
    Human-in-the-loop checkpoint 3.
    Human reviews validation metrics and decides to accept or redo.
    """
    print("\n" + "="*60)
    print("HUMAN CHECKPOINT 3: Accept Correction Results")
    print("="*60)

    metrics = state["validation_metrics"]

    batch_metrics = metrics.get("batch_variance_reduction", {})
    bio_metrics   = metrics.get("biological_signal_preservation", {})

    display = f"""
CORRECTION VALIDATION SUMMARY:
{"─"*40}

1. Batch Variance Reduction:
   η² Before:   {batch_metrics.get('eta_sq_before', 'N/A'):.4f}
   η² After:    {batch_metrics.get('eta_sq_after',  'N/A'):.4f}
   Reduction:   {batch_metrics.get('pct_reduction', 0):.1f}%
   Adequate:    {batch_metrics.get('adequate', 'N/A')}

2. Biological Signal Preservation:
   Markers checked: {bio_metrics.get('markers_checked', [])}"""

    before = bio_metrics.get("signal_before", {})
    after  = bio_metrics.get("signal_after",  {})
    for marker in before:
        if marker in after:
            display += (f"\n   {marker}: effect {before[marker]['effect_size']:.3f}"
                        f" → {after[marker]['effect_size']:.3f} "
                        f"(p: {before[marker]['pvalue']:.3e} → {after[marker]['pvalue']:.3e})")

    display += f"""

Validation plots saved to: results/batch/before_after_pca.png
AI Reasoning: {state['agent_reasoning'][-1]}

DECISION REQUIRED:
  Accept:  proceed to save corrected data
  Reject:  return to strategy selection with new instructions
  Partial: accept but flag concerns for downstream analysis
"""

    human_response = interrupt({
        "checkpoint": "accept_results",
        "message":    display,
        "validation_metrics": metrics,
        "instructions": (
            "Respond with: {'decision': 'accept'|'reject'|'partial', "
            "'notes': 'any notes to attach to output', "
            "'retry_strategy': 'new strategy if rejecting'}"
        )
    })

    if isinstance(human_response, str):
        try:
            response_data = json.loads(human_response)
        except Exception:
            response_data = {"decision": "accept", "notes": human_response}
    else:
        response_data = human_response

    decision = response_data.get("decision", "accept")
    state["human_decisions"].append({
        "checkpoint": "accept_results",
        "input":      display,
        "response":   response_data
    })

    if decision == "reject":
        # Route back — in a full implementation this would loop the graph
        state["final_status"]    = "REJECTED_RETRY"
        state["approved_strategy"]["retry_instructions"] = \
            response_data.get("retry_strategy", "")
        print(f"  Results rejected. Retry instructions: "
              f"{response_data.get('retry_strategy', 'None provided')}")
    elif decision == "partial":
        state["final_status"] = "ACCEPTED_WITH_CAVEATS"
        # Store caveats in approved_strategy dict (state must stay serialisable)
        state["approved_strategy"]["caveats"] = response_data.get("notes", "")
        print(f"  Partial acceptance. Notes: {response_data.get('notes', '')}")
    else:
        state["final_status"] = "ACCEPTED"
        print("  Results accepted.")

    return state


# ================================================================
# NODE 7: SAVE OUTPUTS
# ================================================================

def node_save_outputs(state: BatchCorrectionState) -> BatchCorrectionState:
    """
    Save corrected data, audit trail, and summary report.
    """
    print("\n[NODE 7] Saving outputs...")

    os.makedirs("results/batch", exist_ok=True)

    # Load corrected DataFrame from cache and save as final CSV
    if state.get("df_corrected_path"):
        df_corrected = _load_df(state["df_corrected_path"])
        out_path = "results/batch/olink_batch_corrected.csv"
        df_corrected.to_csv(out_path, index=False)
        print(f"  Corrected data: {out_path}")

    # Save audit trail
    audit = {
        "pipeline":            "Human-in-the-Loop Batch Correction",
        "data_path":           state["data_path"],
        "df_raw_path":         state.get("df_raw_path", ""),
        "df_corrected_path":   state.get("df_corrected_path", ""),
        "confirmed_fields":    state["confirmed_batch_fields"],
        "primary_batch_field": state["primary_batch_field"],
        "batch_assessment":    state["batch_assessment"],
        "approved_strategy":   state["approved_strategy"],
        "validation_metrics":  state["validation_metrics"],
        "human_decisions":     state["human_decisions"],
        "agent_reasoning":     state["agent_reasoning"],
        "final_status":        state["final_status"],
    }
    audit_path = "results/batch/audit_trail.json"
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2, default=str)
    print(f"  Audit trail:    {audit_path}")

    # Summary report
    _write_summary_report(state)

    return state


def _write_summary_report(state: BatchCorrectionState):
    """Write a human-readable markdown summary report."""
    lines = [
        "# Batch Correction Report — UK Biobank OLINK",
        f"\n**Status:** {state['final_status']}",
        f"**Data:** {state['data_path']}",
        f"**Primary batch field:** {state['primary_batch_field']}",
        "\n## Batch Assessment",
    ]

    for field, metrics in state["batch_assessment"].items():
        lines.append(f"\n### {field}")
        lines.append(f"- Proteins affected: {metrics['pct_proteins_affected']:.1f}%")
        lines.append(f"- Mean η²: {metrics['mean_eta_squared']:.4f}")
        lines.append(f"- Severity: {metrics['severity']}")

    lines.append("\n## Correction Strategy")
    strategy = state.get("approved_strategy", {})
    lines.append(f"- Method: {strategy.get('recommended_method')}")
    lines.append(f"- Justification: {strategy.get('justification', '')}")

    vm = state.get("validation_metrics", {})
    bv = vm.get("batch_variance_reduction", {})
    lines.append("\n## Validation")
    lines.append(f"- η² reduction: {bv.get('pct_reduction', 0):.1f}%")
    lines.append(f"- Adequate correction: {bv.get('adequate', 'N/A')}")

    lines.append("\n## Human Decisions")
    for decision in state["human_decisions"]:
        lines.append(f"\n**Checkpoint:** {decision['checkpoint']}")
        lines.append(f"Response: {json.dumps(decision['response'], indent=2)}")

    report_path = "results/batch/correction_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Summary report: {report_path}")


# ================================================================
# ROUTING FUNCTIONS
# ================================================================

def route_after_detection(state: BatchCorrectionState) -> Literal["human_confirm_batch_fields", END]:
    if state.get("final_status") == "ERROR":
        return END
    return "human_confirm_batch_fields"


def route_after_validation(state: BatchCorrectionState) -> Literal["human_accept_results", END]:
    return "human_accept_results"


def route_after_acceptance(state: BatchCorrectionState) -> Literal["node_save_outputs", END]:
    status = state.get("final_status", "ACCEPTED")
    if status == "REJECTED_RETRY":
        return END  # In production, loop back to node_propose_strategy
    return "node_save_outputs"


# ================================================================
# GRAPH CONSTRUCTION
# ================================================================

def build_graph() -> StateGraph:
    """
    Assemble the LangGraph StateGraph with all nodes, edges,
    and human interrupt checkpoints.
    """
    graph = StateGraph(BatchCorrectionState)

    # Add all nodes
    graph.add_node("node_load_data",             node_load_data)
    graph.add_node("node_detect_batch_fields",   node_detect_batch_fields)
    graph.add_node("human_confirm_batch_fields", human_confirm_batch_fields)
    graph.add_node("node_assess_batch_effects",  node_assess_batch_effects)
    graph.add_node("node_propose_strategy",      node_propose_strategy)
    graph.add_node("human_approve_strategy",     human_approve_strategy)
    graph.add_node("node_apply_correction",      node_apply_correction)
    graph.add_node("node_validate_correction",   node_validate_correction)
    graph.add_node("human_accept_results",       human_accept_results)
    graph.add_node("node_save_outputs",          node_save_outputs)

    # Edges
    graph.add_edge(START,                             "node_load_data")
    graph.add_conditional_edges(
        "node_load_data",
        lambda s: END if s.get("final_status") == "ERROR" else "node_detect_batch_fields",
        ["node_detect_batch_fields", END]
    )
    graph.add_edge("node_detect_batch_fields",        "human_confirm_batch_fields")
    graph.add_edge("human_confirm_batch_fields",      "node_assess_batch_effects")
    graph.add_edge("node_assess_batch_effects",       "node_propose_strategy")
    graph.add_edge("node_propose_strategy",           "human_approve_strategy")
    graph.add_edge("human_approve_strategy",          "node_apply_correction")
    graph.add_edge("node_apply_correction",           "node_validate_correction")
    graph.add_edge("node_validate_correction",        "human_accept_results")
    graph.add_conditional_edges("human_accept_results", route_after_acceptance,
                                  ["node_save_outputs", END])
    graph.add_edge("node_save_outputs",               END)

    return graph


def create_pipeline() -> tuple:
    """
    Compile the graph with a MemorySaver checkpointer
    (enables interrupt/resume for human-in-the-loop).
    """
    graph    = build_graph()
    memory   = MemorySaver()
    compiled = graph.compile(checkpointer=memory,
                               interrupt_before=[
                                   "human_confirm_batch_fields",
                                   "human_approve_strategy",
                                   "human_accept_results"
                               ])
    return compiled, memory


# ================================================================
# RUNNER
# ================================================================

def run_pipeline(data_path: str, thread_id: str = "batch_correction_01"):
    """
    Run the full human-in-the-loop batch correction pipeline.
    At each human checkpoint, the pipeline suspends via interrupt(),
    prints the decision context, waits for terminal input, then resumes.
    """
    from langgraph.types import Command   # import here to avoid circular issues

    pipeline, _ = create_pipeline()
    config = {"configurable": {"thread_id": thread_id}}

    # Initial state — only serialisable primitives; no DataFrames
    initial_state = BatchCorrectionState(
        data_path=data_path,
        df_raw_path="", df_current_path="",
        protein_cols=[], metadata_cols=[],
        candidate_batch_fields={}, confirmed_batch_fields=[],
        primary_batch_field="",
        batch_assessment={}, assessment_plots=[],
        correction_needed=False,
        proposed_strategy={}, approved_strategy={},
        df_corrected_path="",
        validation_metrics={}, validation_plots=[],
        human_decisions=[], agent_reasoning=[],
        current_step="init", error_log=[],
        final_status="RUNNING"
    )

    print("\n" + "="*60)
    print("STARTING: Human-in-the-Loop Batch Correction Pipeline")
    print("="*60)

    def _stream_until_interrupt(input_val):
        """Stream events until the graph suspends or finishes."""
        for event in pipeline.stream(input_val, config, stream_mode="values"):
            # stream_mode="values" yields full state snapshots; skip printing here
            pass

        graph_state = pipeline.get_state(config)
        return graph_state

    # --- First run — executes until first interrupt ---
    graph_state = _stream_until_interrupt(initial_state)

    # --- Loop: handle each human checkpoint then resume ---
    while graph_state.next:
        next_node = graph_state.next[0]
        print(f"\n  ⏸  PAUSED before: {next_node}")

        # Extract the interrupt payload
        interrupt_val = _get_interrupt_value(graph_state)

        if interrupt_val and isinstance(interrupt_val, dict):
            print("\n" + "─"*60)
            print(interrupt_val.get("message", ""))
            print("\n" + interrupt_val.get("instructions", ""))
        else:
            print(f"  Waiting at checkpoint: {next_node}")

        human_input = input("\nYour response (press ENTER to accept defaults): ").strip()
        if not human_input:
            human_input = '{"approved": true}'

        # Resume graph with human input
        graph_state = _stream_until_interrupt(Command(resume=human_input))

    final_state = pipeline.get_state(config).values
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE — Status: {final_state.get('final_status', 'DONE')}")
    print(f"{'='*60}")
    return final_state


def _get_interrupt_value(state):
    """Extract interrupt payload from graph state."""
    try:
        tasks = state.tasks
        for task in tasks:
            if hasattr(task, 'interrupts') and task.interrupts:
                return task.interrupts[0].value
    except Exception:
        pass
    return None


if __name__ == "__main__":
    from simulate_data import simulate_batch_data
    simulate_batch_data()
    run_pipeline("data/simulated_batch_olink.csv")
