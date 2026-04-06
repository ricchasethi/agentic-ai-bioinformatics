"""
================================================================
CLI Runner — Human-in-the-Loop Batch Correction
================================================================
Usage:
    # With real data
    python run_pipeline.py --data path/to/olink_data.csv

    # With simulated data (for testing)
    python run_pipeline.py --simulate

    # Resume an interrupted run
    python run_pipeline.py --data path/to/olink_data.csv --thread my_run_01
================================================================
"""

import argparse
import json
import os
import sys

# ── Graceful import with helpful messages ─────────────────────
MISSING_PACKAGES = []
try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command
except ImportError:
    MISSING_PACKAGES.append("langgraph")

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    MISSING_PACKAGES.append("langchain-anthropic")

if MISSING_PACKAGES:
    print("Missing packages. Install with:")
    print(f"  pip install {' '.join(MISSING_PACKAGES)}")
    print("\nRunning in DEMO MODE (no LLM, no LangGraph required)\n")
    DEMO_MODE = True
else:
    DEMO_MODE = False

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def demo_pipeline(data_path: str) -> None:
    """
    Full pipeline in demo mode — no LangGraph/LLM required.
    All human checkpoints use stdin/stdout.
    Identical logic to the agentic version, minus the graph infrastructure.
    """
    print("\n" + "="*60)
    print("DEMO MODE: Batch Correction Pipeline (stdin/stdout)")
    print("="*60)

    # ── Load data ────────────────────────────────────────────
    print("\n[STEP 1] Loading data...")
    df = pd.read_csv(data_path)
    print(f"  Loaded: {df.shape[0]} samples × {df.shape[1]} columns")

    # Detect protein vs metadata columns
    meta_keywords = ['eid', 'case', 'age', 'sex', 'bmi', 'apoe', 'smoking',
                      'education', 'plate', 'batch', 'centre', 'date', 'time',
                      'quality', 'freeze', 'delay', 'well', 'source', '131', '53-']
    protein_cols = [c for c in df.columns
                     if not any(k in c.lower() for k in meta_keywords)
                     and pd.api.types.is_numeric_dtype(df[c])]
    metadata_cols = [c for c in df.columns if c not in protein_cols]
    print(f"  Proteins: {len(protein_cols)} | Metadata: {len(metadata_cols)}")

    # ── Detect batch fields (heuristic in demo mode) ─────────
    print("\n[STEP 2] Detecting batch-relevant fields...")
    batch_keywords = ['plate', 'batch', 'well', 'centre', 'center', 'date',
                       'time', 'quality', 'freeze', 'delay', 'flag']
    bio_keywords   = ['age', 'sex', 'bmi', 'apoe', 'smoking', 'education',
                       'case', 'status', 'eid']

    candidate_fields = {}
    for col in metadata_cols:
        col_lower = col.lower()
        if any(b in col_lower for b in bio_keywords):
            continue
        if any(b in col_lower for b in batch_keywords):
            n_unique = df[col].nunique()
            candidate_fields[col] = {
                "n_unique": n_unique,
                "dtype":    str(df[col].dtype),
                "category": "primary_batch" if "plate" in col_lower or "batch" in col_lower
                              else "secondary_batch"
            }

    print("\n  Detected batch-relevant fields:")
    for col, info in candidate_fields.items():
        print(f"    {col:<30} n_unique={info['n_unique']:<5} [{info['category']}]")

    # ── HUMAN CHECKPOINT 1 ────────────────────────────────────
    print("\n" + "─"*60)
    print("HUMAN CHECKPOINT 1: Confirm Batch Fields")
    print("─"*60)
    print("  Detected fields listed above.")
    print(f"  Suggested primary batch field: "
          f"{next((c for c in candidate_fields if 'plate' in c.lower()), list(candidate_fields.keys())[0] if candidate_fields else 'none')}")
    print("\n  Press ENTER to accept all, or type comma-separated field names to use:")
    user_input = input("  > ").strip()

    if user_input:
        confirmed = [f.strip() for f in user_input.split(",")
                      if f.strip() in df.columns]
    else:
        confirmed = list(candidate_fields.keys())

    print("\n  Enter the PRIMARY batch field for ComBat (for plate-level correction):")
    primary = input("  > ").strip()
    if primary not in df.columns:
        primary = confirmed[0] if confirmed else ""

    print(f"\n  ✓ Confirmed: {confirmed}")
    print(f"  ✓ Primary:   {primary}")

    # ── Assess batch effects ──────────────────────────────────
    print("\n[STEP 3] Assessing batch effects...")
    os.makedirs("results/batch", exist_ok=True)
    assessment = {}

    for batch_field in confirmed:
        if batch_field not in df.columns:
            continue
        is_cat = df[batch_field].dtype == object or df[batch_field].nunique() < 50
        eta_vals, p_vals = [], []

        for prot in protein_cols[:150]:  # sample for speed
            y = df[prot].dropna()
            x = df.loc[y.index, batch_field].dropna()
            y = y.loc[x.index]
            if len(y) < 20:
                continue
            if is_cat:
                groups = [y[x == c].values for c in x.unique() if len(y[x == c]) > 2]
                if len(groups) < 2:
                    continue
                _, p    = stats.f_oneway(*groups)
                grand   = y.mean()
                ss_b    = sum(len(g)*(g.mean()-grand)**2 for g in groups)
                ss_t    = ((y-grand)**2).sum()
                eta_sq  = float(ss_b/ss_t) if ss_t > 0 else 0
            else:
                try:
                    x_num = pd.to_numeric(x, errors='coerce').dropna()
                    y_num = y.loc[x_num.index]
                    if len(x_num) < 10:
                        continue
                    rho, p = stats.spearmanr(x_num, y_num)
                    eta_sq = float(rho**2)
                except Exception:
                    continue
            eta_vals.append(eta_sq)
            p_vals.append(float(p))

        n_sig        = sum(p < 0.05 for p in p_vals)
        pct_affected = n_sig / max(len(protein_cols), 1) * 100
        mean_eta     = float(np.mean(eta_vals)) if eta_vals else 0.0

        if pct_affected > 30 or mean_eta > 0.05:
            severity = "HIGH"
        elif pct_affected > 10 or mean_eta > 0.01:
            severity = "MODERATE"
        else:
            severity = "LOW"

        assessment[batch_field] = {
            "pct_affected": pct_affected,
            "mean_eta_sq":  mean_eta,
            "severity":     severity
        }
        print(f"  {batch_field:<30} {pct_affected:5.1f}% affected | "
              f"η²={mean_eta:.4f} | {severity}")

    # PCA before correction
    _demo_pca_plot(df, protein_cols, primary, "results/batch/pca_before.png",
                    "PCA — Before Batch Correction")

    # ── HUMAN CHECKPOINT 2 ────────────────────────────────────
    print("\n" + "─"*60)
    print("HUMAN CHECKPOINT 2: Approve Correction Strategy")
    print("─"*60)

    max_pct = max((a["pct_affected"] for a in assessment.values()), default=0)
    suggested = "ComBat" if max_pct > 10 else "covariate_only"

    print(f"\n  Based on assessment, AI suggests: {suggested}")
    print(f"  Primary batch variable: {primary}")
    print(f"  Biological covariates to protect: age, sex, AD_case, apoe_e4")
    print("\n  Options:")
    print("    [1] ComBat correction (recommended if severity HIGH/MODERATE)")
    print("    [2] Covariate-only (include batch as regression covariate)")
    print("    [3] No correction")
    choice = input("\n  Enter choice [1/2/3, default=1]: ").strip() or "1"

    method_map = {"1": "ComBat", "2": "covariate_only", "3": "none"}
    method = method_map.get(choice, "ComBat")
    print(f"\n  ✓ Approved method: {method}")

    # ── Apply correction ──────────────────────────────────────
    print(f"\n[STEP 4] Applying correction: {method}...")
    protect_vars = [v for v in ['age', 'sex', 'AD_case', 'apoe_e4']
                     if v in df.columns]

    if method == "ComBat":
        df_corrected = _apply_demo_combat(df, protein_cols, primary, protect_vars)
    else:
        df_corrected = df.copy()
        print("  No data transformation applied (covariate-only mode).")

    # ── Validate ──────────────────────────────────────────────
    print("\n[STEP 5] Validating correction...")
    metrics = _demo_validate(df, df_corrected, protein_cols, primary)

    _demo_pca_plot(df_corrected, protein_cols, primary,
                    "results/batch/pca_after.png",
                    "PCA — After Batch Correction")
    _demo_before_after_plot(df, df_corrected, protein_cols, primary)

    print(f"\n  Batch variance reduction: {metrics['pct_reduction']:.1f}%")
    for marker, vals in metrics.get("marker_effects", {}).items():
        print(f"  {marker}: effect {vals['before']:.3f} → {vals['after']:.3f}")

    # ── HUMAN CHECKPOINT 3 ────────────────────────────────────
    print("\n" + "─"*60)
    print("HUMAN CHECKPOINT 3: Accept Results")
    print("─"*60)
    print(f"\n  η² reduction: {metrics['pct_reduction']:.1f}%")
    print(f"  Plots saved to: results/batch/")
    print("\n  Accept correction results?")
    print("    [1] Accept — save corrected data")
    print("    [2] Accept with caveats — save but flag issues")
    print("    [3] Reject — discard correction")
    decision = input("\n  Enter choice [1/2/3, default=1]: ").strip() or "1"

    decision_map = {"1": "ACCEPTED", "2": "ACCEPTED_WITH_CAVEATS", "3": "REJECTED"}
    final_status = decision_map.get(decision, "ACCEPTED")
    print(f"\n  ✓ Decision: {final_status}")

    # ── Save outputs ──────────────────────────────────────────
    if final_status != "REJECTED":
        print("\n[STEP 6] Saving outputs...")
        out_path = "results/batch/olink_batch_corrected.csv"
        df_corrected.to_csv(out_path, index=False)
        print(f"  Corrected data: {out_path}")

        audit = {
            "status":            final_status,
            "method":            method,
            "primary_batch":     primary,
            "confirmed_fields":  confirmed,
            "assessment":        assessment,
            "validation":        metrics
        }
        with open("results/batch/audit_trail.json", "w") as f:
            json.dump(audit, f, indent=2, default=str)
        print("  Audit trail: results/batch/audit_trail.json")

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE — Status: {final_status}")
    print(f"{'='*60}\n")


def _apply_demo_combat(df, protein_cols, batch_col, protect_vars):
    """Demo ComBat: mean-centering per batch (stand-in for real pyComBat)."""
    df_corrected = df.copy()
    if batch_col not in df.columns:
        return df_corrected

    # Impute first
    imputer = SimpleImputer(strategy='median')
    df_corrected[protein_cols] = imputer.fit_transform(df_corrected[protein_cols])

    grand_means = df_corrected[protein_cols].mean()
    batches = df_corrected[batch_col].unique()

    for batch_val in batches:
        mask = df_corrected[batch_col] == batch_val
        n    = mask.sum()
        if n < 2:
            continue
        batch_means = df_corrected.loc[mask, protein_cols].mean()
        shift = grand_means - batch_means
        df_corrected.loc[mask, protein_cols] = \
            df_corrected.loc[mask, protein_cols].add(shift)

    print(f"  Applied mean-centering per batch ({len(batches)} batches).")
    print("  NOTE: In production use pyComBat (pip install inmoose) for full ComBat.")
    return df_corrected


def _demo_validate(df_raw, df_corrected, protein_cols, batch_col):
    """Compute validation metrics."""
    eta_before, eta_after = [], []
    marker_effects = {}
    known_markers = ['NEFL', 'GFAP', 'TREM2', 'CLU', 'APOE', 'APP']

    for prot in protein_cols[:100]:
        for df_, store in [(df_raw, eta_before), (df_corrected, eta_after)]:
            if batch_col not in df_.columns:
                continue
            y = df_[prot].dropna()
            x = df_.loc[y.index, batch_col].dropna()
            y = y.loc[x.index]
            groups = [y[x==c].values for c in x.unique() if len(y[x==c]) > 1]
            if len(groups) < 2:
                store.append(0.0)
                continue
            grand = y.mean()
            ss_b  = sum(len(g)*(g.mean()-grand)**2 for g in groups)
            ss_t  = ((y-grand)**2).sum()
            store.append(float(ss_b/ss_t) if ss_t > 0 else 0.0)

    pct_reduction = (1 - np.mean(eta_after)/max(np.mean(eta_before), 1e-9)) * 100

    if 'AD_case' in df_raw.columns:
        for marker in known_markers:
            if marker not in df_raw.columns:
                continue
            effects = {}
            for label, df_ in [("before", df_raw), ("after", df_corrected)]:
                cases    = df_.loc[df_['AD_case']==1, marker].dropna()
                controls = df_.loc[df_['AD_case']==0, marker].dropna()
                if len(cases) > 5 and len(controls) > 5:
                    effects[label] = float(
                        (cases.mean()-controls.mean()) / df_[marker].std()
                    )
            if len(effects) == 2:
                marker_effects[marker] = effects

    return {
        "eta_sq_before":  float(np.mean(eta_before)),
        "eta_sq_after":   float(np.mean(eta_after)),
        "pct_reduction":  float(pct_reduction),
        "marker_effects": marker_effects
    }


def _demo_pca_plot(df, protein_cols, colour_col, save_path, title):
    """PCA coloured by batch variable."""
    try:
        valid = [p for p in protein_cols if p in df.columns][:200]
        X = SimpleImputer(strategy='median').fit_transform(df[valid])
        X = StandardScaler().fit_transform(X)
        pcs = PCA(n_components=2, random_state=42).fit_transform(X)

        fig, ax = plt.subplots(figsize=(8, 6))
        if colour_col and colour_col in df.columns and df[colour_col].nunique() <= 20:
            for i, val in enumerate(df[colour_col].unique()):
                mask = df[colour_col].values == val
                ax.scatter(pcs[mask, 0], pcs[mask, 1], alpha=0.4, s=12, label=str(val))
            if df[colour_col].nunique() <= 8:
                ax.legend(title=colour_col, fontsize=7)
        else:
            ax.scatter(pcs[:, 0], pcs[:, 1], alpha=0.4, s=12, color='steelblue')

        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title(title); plt.tight_layout()
        plt.savefig(save_path, dpi=130, bbox_inches='tight')
        plt.close()
        print(f"  Plot saved: {save_path}")
    except Exception as e:
        print(f"  Plot failed: {e}")


def _demo_before_after_plot(df_raw, df_corrected, protein_cols, batch_col):
    """Side-by-side before/after PCA."""
    try:
        valid = [p for p in protein_cols if p in df_raw.columns][:200]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, df_, title in [
            (axes[0], df_raw,       "Before Correction"),
            (axes[1], df_corrected, "After Correction")
        ]:
            X = SimpleImputer(strategy='median').fit_transform(df_[valid])
            X = StandardScaler().fit_transform(X)
            pcs = PCA(n_components=2, random_state=42).fit_transform(X)

            if batch_col and batch_col in df_.columns:
                unique_batches = df_[batch_col].unique()
                cmap = plt.cm.get_cmap('tab20', min(len(unique_batches), 20))
                for i, bv in enumerate(unique_batches[:20]):
                    mask = df_[batch_col].values == bv
                    ax.scatter(pcs[mask, 0], pcs[mask, 1],
                                color=cmap(i % 20), alpha=0.4, s=10,
                                label=str(bv) if i < 6 else "")
            else:
                ax.scatter(pcs[:, 0], pcs[:, 1], alpha=0.4, s=10, color='steelblue')

            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

        plt.suptitle("Batch Effect Correction: Before vs After", fontsize=14)
        plt.tight_layout()
        path = "results/batch/before_after_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Comparison plot: {path}")
    except Exception as e:
        print(f"  Comparison plot failed: {e}")


# ── CLI ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Human-in-the-Loop Batch Correction — UK Biobank OLINK"
    )
    parser.add_argument("--data",     type=str, help="Path to OLINK CSV file")
    parser.add_argument("--simulate", action="store_true",
                         help="Generate and use simulated data")
    parser.add_argument("--thread",   type=str, default="batch_run_01",
                         help="Thread ID for resuming interrupted runs")
    args = parser.parse_args()

    if args.simulate or not args.data:
        print("Generating simulated UK Biobank OLINK data...")
        from simulate_data import simulate_batch_data
        simulate_batch_data()
        data_path = "data/simulated_batch_olink.csv"
    else:
        data_path = args.data

    if not os.path.exists(data_path):
        print(f"Error: data file not found: {data_path}")
        sys.exit(1)

    if DEMO_MODE:
        print("Running in DEMO MODE (no LLM, no LangGraph).")
        demo_pipeline(data_path)
    else:
        # Full LangGraph agentic pipeline
        from batch_correction_agent import run_pipeline
        run_pipeline(data_path, thread_id=args.thread)


if __name__ == "__main__":
    main()
