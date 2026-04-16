# Batch Correction Report — UK Biobank OLINK

**Status:** ACCEPTED
**Data:** data/simulated_batch_olink.csv
**Primary batch field:** plate_id

## Batch Assessment

### plate_id
- Proteins affected: 74.9%
- Mean η²: 0.1975
- Severity: HIGH — ComBat correction required

### well_position
- Proteins affected: 3.0%
- Mean η²: 0.0003
- Severity: LOW — covariate adjustment sufficient

### assessment_centre
- Proteins affected: 71.4%
- Mean η²: 0.0376
- Severity: HIGH — ComBat correction required

### collection_date
- Proteins affected: 3.5%
- Mean η²: 0.1659
- Severity: HIGH — ComBat correction required

### collection_time_mins
- Proteins affected: 6.5%
- Mean η²: 0.0004
- Severity: LOW — covariate adjustment sufficient

### processing_delay_mins
- Proteins affected: 43.7%
- Mean η²: 0.0022
- Severity: HIGH — ComBat correction required

### freeze_thaw_cycles
- Proteins affected: 20.6%
- Mean η²: 0.0019
- Severity: MODERATE — ComBat recommended

### sample_quality_flag
- Proteins affected: 38.7%
- Mean η²: 0.0024
- Severity: HIGH — ComBat correction required

## Reflection Agent

1 reflection round(s) completed.

**Final plan:** After reflection, 6 field(s) retain sufficient batch signal for ComBat (plate_id, assessment_centre, collection_date, processing_delay_mins, freeze_thaw_cycles, sample_quality_flag). 2 demoted to covariate (well_position, collection_time_mins). 0 dropped (none).

**Dropped fields:** []

**Demoted to covariate:** ['well_position', 'collection_time_mins']

**ComBat fields:** ['plate_id', 'assessment_centre', 'collection_date', 'processing_delay_mins', 'freeze_thaw_cycles', 'sample_quality_flag']

### Round 1
- plate_id: KEEP_PRIMARY (HIGH) — η²=0.1983, 99.5% proteins — HIGH. Primary ComBat variable confirmed.
- well_position: DEMOTE_TO_COVARIATE (LOW) — η²=0.0003, 3.0% proteins — LOW. ComBat would risk over-correction; include 'well_position' as regression covariate only.
- assessment_centre: KEEP_SECONDARY (HIGH) — η²=0.0353, 95.0% proteins — HIGH. Retain as secondary ComBat covariate.
- collection_date: KEEP_SECONDARY (HIGH) — η²=0.1661, 4.5% proteins — HIGH. Retain as secondary ComBat covariate.
- collection_time_mins: DEMOTE_TO_COVARIATE (LOW) — η²=0.0004, 6.5% proteins — LOW. ComBat would risk over-correction; include 'collection_time_mins' as regression covariate only.
- processing_delay_mins: KEEP_SECONDARY (HIGH) — η²=0.0023, 56.8% proteins — HIGH. Retain as secondary ComBat covariate.
- freeze_thaw_cycles: KEEP_SECONDARY (MODERATE) — η²=0.0020, 25.6% proteins — MODERATE. Retain as secondary batch covariate.
- sample_quality_flag: KEEP_SECONDARY (HIGH) — η²=0.0024, 54.3% proteins — HIGH. Retain as secondary ComBat covariate.

*LLM critique:* The automated decisions appear sound overall, with appropriate retention of the primary batch variable (plate_id, η²=0.198) and key technical covariates showing substantial protein effects. However, I have concerns about retaining **collection_date** as a ComBat covariate despite affecting only 4.5% of proteins - while its high η²=0.166 suggests strong effects when present, the limited protein coverage may indicate this represents genuine biological variation (e.g., seasonal immune patterns) rather than technical batch effects that should be corrected. The demotion of well_position and collection_time_mins to regression covariates is appropriate given their low effect sizes, preventing over-correction while still accounting for spatial/temporal gradients. For AD biomarker discovery, this plan should effectively remove major technical artifacts while preserving biological signal, though careful validation of collection_date effects is recommended to ensure we're not inadvertently removing relevant temporal biology.

## Correction Strategy
- Method: ComBat
- Justification: plate_id shows severe batch effects (η²=0.198, 75% proteins affected) requiring ComBat correction. Five additional variables show significant batch signals: assessment_centre (η²=0.038, 71% proteins), collection_date (η²=0.166, 4% proteins but high effect size), processing_delay_mins (η²=0.002, 44% proteins), sample_quality_flag (η²=0.002, 39% proteins), and freeze_thaw_cycles (η²=0.002, 21% proteins). Sequential ComBat correction recommended starting with plate_id as primary batch variable. well_position and collection_time_mins have low severity and can be handled as covariates in downstream analysis.

## Validation
- η² reduction: 99.9%
- Adequate correction: True

## Human Decisions

**Checkpoint:** confirm_batch_fields
Response: "{\"approved\": true}"

**Checkpoint:** approve_strategy
Response: {
  "approved": true
}

**Checkpoint:** accept_results
Response: {
  "approved": true
}