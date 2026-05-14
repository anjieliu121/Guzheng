# Evaluation Report

## Training Data Baseline
- **n_files:** 133
- **mean_n_notes:** 342.6
- **mean_duration:** 0.5854
- **mean_density:** 2.58
- **mean_velocity:** 79.2
- **mean_pc_entropy:** 2.362
- **mean_penta_purity:** 0.9983
- **mean_large_leap_rate:** 0.0989
- **mean_interval:** 6.23
- **mean_max_simul:** 2.2
- **mean_pitch:** 64.1

## Test Data
- **n_files:** 20
- **mean_n_notes:** 233.9
- **mean_duration:** 0.5541
- **mean_density:** 2.32
- **mean_velocity:** 79.2
- **mean_pc_entropy:** 2.317
- **mean_penta_purity:** 0.9982
- **mean_large_leap_rate:** 0.0684
- **mean_interval:** 6.39
- **mean_max_simul:** 2.0
- **mean_pitch:** 65.0

## Generated Variants

| Metric | Training | constrained | unconstrained |
|--------|----------|----------|----------|
| mean_n_notes | 342.6 | 481.4 | 491.1 |
| mean_density | 2.58 | 8.04 | 4.41 |
| mean_duration | 0.5854 | 0.2772 | 0.3913 |
| mean_velocity | 79.2 | 73.4 | 74.7 |
| mean_pc_entropy | 2.362 | 1.881 | 1.744 |
| mean_penta_purity | 0.9983 | 1.0 | 1.0 |
| mean_large_leap_rate | 0.0989 | 0.113 | 0.1116 |
| mean_interval | 6.23 | 6.77 | 5.91 |
| mean_max_simul | 2.2 | 3.6 | 3.7 |
| mean_pitch | 64.1 | 60.7 | 63.2 |

## OA Metrics (1.0 = identical to training)

| Distribution | constrained | unconstrained |
|-------------|----------|----------|
| OA_pitch_class | 0.7336 | 0.7381 |
| OA_duration | 0.5076 | 0.3889 |
| OA_interval | 0.7969 | 0.7043 |
| OA_ioi | 0.4016 | 0.3381 |
