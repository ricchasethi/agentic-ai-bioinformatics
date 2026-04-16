# UK Biobank OLINK Batch Correction

A human-in-the-loop agentic pipeline for batch effect detection and correction of UK Biobank OLINK proteomics data, built for Alzheimer's Disease biomarker discovery. Powered by [LangGraph](https://github.com/langchain-ai/langgraph) and Claude claude-sonnet-4-20250514. We generate synthetic OLINK data and test this workflow on it.

---

## Overview

OLINK proximity extension assays measure protein abundance as NPX (Normalised Protein eXpression) values. When run at scale across a biobank, systematic technical variation — plate effects, centre differences, processing delays — can dwarf the biological signal of interest. This pipeline detects, quantifies, and corrects those batch effects while preserving case-control differences for downstream AD biomarker analysis.

The pipeline is **human-in-the-loop**: an AI agent handles detection and strategy proposal, but a researcher reviews and approves decisions at three critical checkpoints before any data is modified.

A **reflection agent** (`node_reflect_on_assessment`) adds a self-correcting loop between batch assessment and strategy proposal. It automatically prunes fields with no meaningful batch effect, demotes marginal fields to regression covariates, and pre-fills a validated correction strategy — so the human checkpoint always shows a clean, evidence-backed plan rather than raw LLM output.

---

## Pipeline Architecture

```
START
  │
  ▼
[1] node_load_data              — load CSV, classify protein vs. metadata columns
  │
  ▼
[2] node_detect_batch_fields    — LLM classifies metadata columns by batch relevance
  │
  ⏸ CHECKPOINT 1: human_confirm_batch_fields
  │
  ▼
[3] node_assess_batch_effects   — compute η² per field, generate PCA plots
  │
  ▼
[4] node_reflect_on_assessment  — ★ reflection agent: prune / demote / drop fields
  │                                 with no batch signal; pre-fill strategy prior
  ▼
[5] node_propose_strategy       — LLM confirms / refines the reflection prior
  │
  ⏸ CHECKPOINT 2: human_approve_strategy
  │
  ▼
[6] node_apply_correction       — pyComBat or covariate-only correction
  │
  ▼
[7] node_validate_correction    — η² reduction, AD marker signal preservation
  │
  ⏸ CHECKPOINT 3: human_accept_results
  │
  ▼
[8] node_save_outputs           — corrected CSV, audit trail, markdown report
  │
  ▼
 END
```

**Error path:** `node_load_data` → `END` if `final_status == "ERROR"`  
**Reject path:** `human_accept_results` → `END` if `final_status == "REJECTED_RETRY"`

---

## Reflection Agent

`node_reflect_on_assessment` runs automatically after `node_assess_batch_effects`
and before any strategy is proposed. It removes the need for the LLM or the human
to manually filter out spurious batch fields.

### What it does

For each confirmed batch field it applies the same thresholds used during
assessment:

| η² | % proteins | Severity | Decision | Outcome |
|---|---|---|---|---|
| ≥ 0.05 | ≥ 30% | HIGH | KEEP | Enters ComBat as primary/secondary |
| ≥ 0.01 | ≥ 10% | MODERATE | KEEP | Enters ComBat |
| ≥ 0.001 | ≥ 2% | LOW | DEMOTE | Regression covariate only; no ComBat |
| < 0.001 | < 2% | NONE | DROP | Excluded entirely |

If no field survives the KEEP threshold, the overall strategy automatically
falls back to `covariate_only` or `none`, and `correction_needed` is set to
`False`.

### Reflection loop

The agent iterates up to three times (configurable via `_MAX_REFLECT_ROUNDS`):

1. Classify each confirmed field.
2. Build a revised field list and recommended method.
3. Ask the LLM to critique the decisions (flags any incorrect demotions or
   missed risks for AD biomarker analysis).
4. Re-assess surviving fields on the current DataFrame to confirm their
   η² still holds.
5. Stop early if the plan is stable between consecutive rounds.

The full per-round log is stored in `state["reflection_log"]` and written to
`audit_trail.json` and `correction_report.md`.

### What the human sees at Checkpoint 2

The strategy approval screen now includes a **Reflection Agent Summary**
section listing each field's final decision (KEEP / DEMOTE / DROP) and the
reason, alongside the proposed correction method. Dropped and demoted fields
are shown explicitly so the researcher can override if domain knowledge warrants it.

### Running standalone

The reflection logic can be run independently without LangGraph for debugging:

```bash
python batch_correction_reflection_agent.py data/simulated_batch_olink.csv
```

---

## Human Checkpoints

| Checkpoint | What the human decides |
|---|---|
| **1 — Confirm batch fields** | Review AI-detected batch-relevant columns; select primary batch variable for ComBat |
| **2 — Approve strategy** | Review reflection summary (dropped/demoted fields) and proposed method; accept, modify, or reject |
| **3 — Accept results** | Review η² reduction and AD marker preservation; accept, flag caveats, or reject |

---

## Quickstart

### Prerequisites

```bash
pip install langgraph langchain langchain-anthropic langchain-core
pip install pandas numpy scipy scikit-learn matplotlib seaborn
pip install inmoose pyarrow   # pyComBat + parquet cache
```

Or install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Run with simulated data (no API key required)

```bash
python run_pipeline.py --simulate
```

This generates a synthetic UK Biobank dataset (`data/simulated_batch_olink.csv`) with realistic plate effects, centre effects, and AD case-control signal, then runs the full demo pipeline via stdin/stdout.

### Run with real data

```bash
export ANTHROPIC_API_KEY=your_key_here
python run_pipeline.py --data path/to/your_olink_data.csv
```

### Resume an interrupted run

```bash
python run_pipeline.py --data your_olink_data.csv --thread my_run_01
```

Thread IDs must match the original run for LangGraph's `MemorySaver` to restore the correct checkpoint.

---

## File Structure

```
batch_correction/
├── CLAUDE.md                            ← project constraints and conventions
├── batch_correction_agent.py            ← LangGraph graph, all nodes, state
│                                           (reflection node fully integrated)
├── batch_correction_reflection_agent.py ← standalone reflection module
├── run_pipeline.py                      ← CLI entry point, demo mode fallback
├── simulate_data.py                     ← synthetic UKB OLINK data generator
├── requirements.txt
├── data/
│   └── simulated_batch_olink.csv        ← generated by simulate_data.py
└── results/
    └── batch/
        ├── cache/                       ← TRANSIENT: parquet cache (do not commit)
        ├── olink_batch_corrected.csv
        ├── audit_trail.json             ← includes reflection_log
        ├── correction_report.md         ← includes reflection section
        ├── pca_before.png
        ├── pca_after.png
        └── before_after_comparison.png
```

> **Note:** `results/batch/cache/` is a runtime cache for intermediate DataFrames. It is transient and should never be committed. The canonical outputs are `olink_batch_corrected.csv` and `audit_trail.json`.

---

## Outputs

| File | Description |
|---|---|
| `olink_batch_corrected.csv` | Final corrected protein NPX matrix with metadata |
| `audit_trail.json` | Complete record of batch assessment, reflection log, strategy, validation, and human decisions |
| `correction_report.md` | Human-readable summary including reflection agent decisions |
| `pca_before.png` / `pca_after.png` | PCA coloured by primary batch field, before and after correction |
| `before_after_comparison.png` | Side-by-side PCA comparison |

---

## Batch Severity Thresholds

Used identically by `node_assess_batch_effects` and the reflection agent.
Do not change one without changing the other.

| Mean η² | Proteins affected | Severity | Reflection decision | Correction action |
|---|---|---|---|---|
| ≥ 0.05 | ≥ 30% | High | KEEP | ComBat required |
| ≥ 0.01 | ≥ 10% | Moderate | KEEP | ComBat recommended |
| ≥ 0.001 | ≥ 2% | Low | DEMOTE to covariate | Include in regression only |
| < 0.001 | < 2% | None | DROP | Exclude entirely |

---

## Correction Methods

### ComBat (default for moderate/high severity)

Uses [`inmoose.pycombat.pycombat_norm`](https://github.com/epigenelabs/inmoose), an empirical Bayes method that models and removes additive and multiplicative batch effects. Only applied to fields the reflection agent classifies as KEEP.

**Critical:** biological covariates (`age`, `sex`, `AD_case`, `apoe_e4`) are always passed as `covar_mod` to prevent ComBat from treating case-control differences as batch noise.

```python
corrected = pycombat_norm(
    protein_matrix.T,     # pyComBat expects proteins × samples
    batch_labels,
    covar_mod=covar_df.T  # protect biological signal
)
```

### Covariate-only (for low severity, or when reflection demotes all fields)

No transformation is applied to the data. Surviving batch fields are retained as covariates for inclusion in downstream regression models.

---

## QC Processing Order

The order of steps is strict and must not be changed:

1. **Missingness filtering** — remove samples/proteins with > 20% missing values  
2. **Outlier winsorisation** — clip at ± 4 SD per protein  
3. **Imputation** — median imputation for residual NAs  
4. **Batch effect detection** — assess η² on the clean, complete matrix  
5. **Reflection** — prune fields with no batch signal before strategy is set  
6. **Batch correction (ComBat)** — correct on pre-INT values  
7. **Rank-based INT** — apply inverse normal transformation *after* correction  
8. **Association testing** — logistic regression or Cox proportional hazards  

---

## UK Biobank Field Conventions

| Field | Description |
|---|---|
| `53-0.0` | Date of attending assessment centre (blood draw date) |
| `131036` | Date G30 (Alzheimer's disease) first reported |
| `131037` | Source of report of G30 |
| `131032` | Date F00 (dementia in Alzheimer's disease) first reported |
| `131033` | Source of report of F00 |
| `plate_id` | OLINK measurement plate — primary ComBat batch variable |
| `assessment_centre` | UKB centre code — secondary batch variable |

### AD case definition

Incident cases only: ICD-10 G30.x or F00.x, diagnosed **after** blood draw date. Prevalent cases (diagnosed before blood draw) are excluded to avoid reverse causation.

Reliable sources: HES inpatient (11), HES outpatient (51), GP records (61). Self-report (31) is excluded from the primary case definition.

---

## LangGraph State

`BatchCorrectionState` is a `TypedDict`. All fields must be plain Python
primitives (no DataFrames, no numpy types). The reflection agent adds one new
field:

```python
reflection_log: list[dict]   # per-round audit trail from the reflection node
```

Always initialise `reflection_log=[]` in `initial_state`. Each entry records
the round number, per-field decisions, the revised plan, and the LLM critique.

For the full msgpack serialisation rules, the `_sanitise()` helper, and the
parquet cache pattern, see **CLAUDE.md §Critical constraints**.

---

## LLM Configuration

| Setting | Value |
|---|---|
| Model | `claude-sonnet-4-20250514` |
| Temperature | `0.1` (low, for analytical consistency) |
| Max tokens | `2048` |

LLM nodes: `node_detect_batch_fields`, `node_reflect_on_assessment` (critique
only), `node_propose_strategy`. All have heuristic fallbacks — the pipeline
always completes in demo mode without an API key.

---

## Simulated Data

`simulate_data.py` generates a realistic 3,000-sample × 200-protein dataset with:

- **15 plates** with additive mean-shift plate effects (σ = 0.8 NPX)
- **3 assessment centres** with centre-level offsets
- **Well position edge effects** (positions 1–8 and 89–96)
- **Processing delay degradation** (negative correlation with NPX)
- **Freeze-thaw cycle degradation**
- **Sample quality flags** (haemolysis, lipaemia)
- **AD biological signal** on 10 known markers (NEFL, GFAP, TREM2, APP, CLU, CR1, BIN1, PICALM, APOE, ADAM10) with effect size ~0.7 NPX

The reflection agent is expected to KEEP `plate_id` (HIGH effect), KEEP or
DEMOTE `assessment_centre`, and DEMOTE or DROP `freeze_thaw_cycles` and
`sample_quality_flag` (LOW/NONE effect in the simulated data).

---

## Known Issues

**pyComBat covariate matrix warning:** pyComBat may print `[WARNING] The covariate matrix seems to be transposed.` This fires when the matrix dimensions are non-square and can be safely ignored — the computation is correct.

**Small batches:** ComBat requires ≥ 3 samples per batch for variance estimation. Batches below this threshold are removed before correction. If many batches are small after QC filtering, consider merging by collection date before running.

**Parquet dtype drift:** Reloading from the parquet cache may change dtype slightly (e.g. `Int64` → `float64` where NAs are present). This is expected pandas/parquet behaviour and does not affect downstream analysis.

**Reflection convergence:** The loop exits when classifications are stable or after `_MAX_REFLECT_ROUNDS` (3). If all three rounds run without converging, verify the protein sample size (`sample_n=150` by default) is sufficient for stable η² estimates at your dataset size.

---

## Dependencies

```
langgraph>=0.2.0
langchain>=0.3.0
langchain-anthropic>=0.3.0
langchain-core>=0.3.0
inmoose>=0.4.0
pandas>=2.0.0
numpy>=1.26.0
scipy>=1.11.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyarrow>=14.0.0
lifelines>=0.27.0    # optional: Cox regression
```

---

## Author

Riccha Sethi — [ricchasethi@gmail.com](mailto:ricchasethi@gmail.com)
