# Data Integrity & Real-Observation Requirements

## Core Principle
OceanEmbed training and validation datasets must contain only legitimate, traceable real-world observations — not synthetic data.

## Allowed Data Sources
- Argo Float Observations (TEMP_QC/PRES_QC flags in ['1','2'])
- Satellite/In-Situ Surface Data (NASA JPL MUR SST, NOAA NESDIS SLA, ESA SMOS SSS)

## Not Allowed
- Synthetic data (generate_data.py output)
- Simulated/modeled data
- Randomly generated numbers
- Manually fabricated test values

## Pre-Training Validation
```bash
python scripts/validate_dataset.py