"""
Simulate realistic UK Biobank OLINK data with batch effects
for testing the human-in-the-loop pipeline.
"""

import numpy as np
import pandas as pd
import os


def simulate_batch_data(
    n_participants: int = 3000,
    n_proteins: int = 200,
    n_plates: int = 15,
    seed: int = 42,
    output_path: str = "data/simulated_batch_olink.csv"
) -> pd.DataFrame:
    """
    Simulate OLINK data with realistic batch structure:
    - Plate effects (additive mean shift per plate)
    - Well position effects (edge effect)
    - Centre effects (3 assessment centres)
    - Collection date drift (linear trend over time)
    - Sample quality variation (haemolysis, processing delay)
    - Biological signal: AD case-control differences
    """
    rng = np.random.default_rng(seed)
    os.makedirs("data", exist_ok=True)

    # ── Protein names ─────────────────────────────────────────
    known_ad = ['NEFL', 'GFAP', 'TREM2', 'APP', 'CLU',
                'CR1', 'BIN1', 'PICALM', 'APOE', 'ADAM10']
    other_proteins = [f"PROT_{i:04d}" for i in range(n_proteins - len(known_ad))]
    protein_names  = known_ad + other_proteins

    # ── Participant metadata ───────────────────────────────────
    centres = rng.choice([11012, 11020, 11035], n_participants, p=[0.4, 0.35, 0.25])
    plates  = rng.choice(range(1, n_plates + 1), n_participants)

    # Well positions (1–96 per plate, 1–8=edges)
    well_pos = rng.integers(1, 97, n_participants)
    is_edge  = (well_pos <= 8) | (well_pos >= 89)

    # Collection dates (2006–2010 initial recruitment)
    base_date   = pd.Timestamp("2006-01-01")
    collect_days= rng.integers(0, 365*4, n_participants)
    collect_date= [base_date + pd.Timedelta(days=int(d)) for d in collect_days]

    # Processing delay (minutes, 30–240)
    proc_delay   = rng.integers(30, 241, n_participants)
    # Freeze-thaw cycles (0–3)
    freeze_thaw  = rng.choice([0, 1, 2, 3], n_participants, p=[0.5, 0.3, 0.15, 0.05])
    # Sample quality (1=good, 2=haemolysis, 3=lipaemic)
    sample_qual  = rng.choice([1, 2, 3], n_participants, p=[0.85, 0.10, 0.05])

    # Case/control (~5% AD)
    ad_case  = rng.binomial(1, 0.05, n_participants)
    age      = rng.normal(65, 8, n_participants).clip(40, 85)
    sex      = rng.binomial(1, 0.5, n_participants)
    bmi      = rng.normal(27, 5, n_participants).clip(15, 50)
    apoe_e4  = rng.binomial(1, 0.25, n_participants)

    # ── Protein NPX matrix (base: N(8, 1.5)) ──────────────────
    npx = rng.normal(8.0, 1.5, (n_participants, n_proteins))

    # 1. Plate batch effect (additive, varies per protein)
    plate_effects = rng.normal(0, 0.8, (n_plates, n_proteins))
    for i, plate in enumerate(plates):
        npx[i] += plate_effects[plate - 1]

    # 2. Well edge effect (small additive)
    npx[is_edge] += rng.normal(0.15, 0.05, (is_edge.sum(), n_proteins))

    # 3. Centre effect
    centre_map = {11012: 0, 11020: 1, 11035: 2}
    centre_effects = rng.normal(0, 0.4, (3, n_proteins))
    for i, c in enumerate(centres):
        npx[i] += centre_effects[centre_map[c]]

    # 4. Processing delay degradation (negative correlation)
    delay_norm = (proc_delay - proc_delay.mean()) / proc_delay.std()
    for j in range(n_proteins):
        npx[:, j] -= delay_norm * rng.uniform(0.0, 0.15)

    # 5. Freeze-thaw degradation
    npx -= freeze_thaw[:, np.newaxis] * rng.uniform(0, 0.1, n_proteins)

    # 6. Sample quality (haemolysis/lipaemia)
    qual_mask = sample_qual > 1
    npx[qual_mask] += rng.normal(0.2, 0.1, (qual_mask.sum(), n_proteins))

    # 7. Biological AD signal on known markers
    case_idx = np.where(ad_case == 1)[0]
    for j, prot in enumerate(known_ad):
        effect = rng.normal(0.7, 0.2)
        npx[case_idx, j] += effect

    # Add measurement noise
    npx += rng.normal(0, 0.1, npx.shape)

    # ── Build DataFrame ────────────────────────────────────────
    protein_df = pd.DataFrame(npx, columns=protein_names)

    meta_df = pd.DataFrame({
        'eid':                  range(1000001, 1000001 + n_participants),
        'AD_case':              ad_case,
        'age':                  age.round(1),
        'sex':                  sex,
        'bmi':                  bmi.round(1),
        'apoe_e4':              apoe_e4,
        # Batch fields (UKB-style naming)
        'plate_id':             plates,
        'well_position':        well_pos,
        'assessment_centre':    centres,
        'collection_date':      [d.strftime('%Y-%m-%d') for d in collect_date],
        'collection_time_mins': rng.integers(480, 1080, n_participants),  # 8am–6pm
        'processing_delay_mins':proc_delay,
        'freeze_thaw_cycles':   freeze_thaw,
        'sample_quality_flag':  sample_qual,
        # UKB field-style names
        '131036':               [None] * n_participants,  # G30 date (set for cases)
        '131037':               [None] * n_participants,  # G30 source
        '53-0.0':               [d.strftime('%Y-%m-%d') for d in collect_date],
    })

    # Set AD diagnosis dates for cases (1–5 years after blood draw)
    for i in case_idx:
        years_after = rng.integers(1, 6)
        diag_date   = collect_date[i] + pd.Timedelta(days=365 * years_after)
        meta_df.at[i, '131036'] = diag_date.strftime('%Y-%m-%d')
        meta_df.at[i, '131037'] = rng.choice([11, 51, 61])  # HES/outpat/GP

    df = pd.concat([meta_df, protein_df], axis=1)

    df.to_csv(output_path, index=False)
    print(f"Simulated data saved: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Plates: {n_plates} | Centres: 3")
    print(f"  Cases: {ad_case.sum()} | Controls: {(ad_case==0).sum()}")
    print(f"  Batch fields included: plate_id, well_position, assessment_centre,")
    print(f"    collection_date, processing_delay_mins, freeze_thaw_cycles, sample_quality_flag")

    return df


if __name__ == "__main__":
    simulate_batch_data()
