# Evaluation Report

## Training Data Baseline
- **n_files:** 147
- **mean_n_notes:** 663.3
- **mean_duration:** 0.569
- **mean_density:** 3.91
- **mean_velocity:** 79.9
- **mean_pc_entropy:** 2.391
- **mean_penta_purity:** 0.9945
- **mean_large_leap_rate:** 0.1215
- **mean_interval:** 6.75
- **mean_max_simul:** 3.6
- **mean_pitch:** 64.1
- **mean_self_rep_4gram:** 0.5576
- **mean_self_rep_8gram:** 0.3001
- **mean_self_rep_12gram:** 0.222

## Test Data
- **n_files:** 22
- **mean_n_notes:** 424.2
- **mean_duration:** 0.5258
- **mean_density:** 3.88
- **mean_velocity:** 80.0
- **mean_pc_entropy:** 2.32
- **mean_penta_purity:** 0.9938
- **mean_large_leap_rate:** 0.0742
- **mean_interval:** 6.47
- **mean_max_simul:** 2.6
- **mean_pitch:** 65.1
- **mean_self_rep_4gram:** 0.7477
- **mean_self_rep_8gram:** 0.5837
- **mean_self_rep_12gram:** 0.519

## Generated Variants

| Metric | Training | constrained | unconstrained |
|--------|----------|----------|----------|
| mean_n_notes | 663.3 | 500.3 | 491.7 |
| mean_density | 3.91 | 6.43 | 7.45 |
| mean_duration | 0.569 | 0.3902 | 0.325 |
| mean_velocity | 79.9 | 80.1 | 79.7 |
| mean_pc_entropy | 2.391 | 2.268 | 2.322 |
| mean_penta_purity | 0.9945 | 1.0 | 0.9883 |
| mean_large_leap_rate | 0.1215 | 0.1819 | 0.2013 |
| mean_interval | 6.75 | 8.01 | 8.41 |
| mean_max_simul | 3.6 | 4.8 | 5.0 |
| mean_pitch | 64.1 | 61.7 | 62.0 |
| mean_self_rep_4gram | 0.5576 | 0.13 | 0.1093 |
| mean_self_rep_8gram | 0.3001 | 0.0014 | 0.0006 |
| mean_self_rep_12gram | 0.222 | 0.0006 | 0.0 |

## OA Metrics (1.0 = identical to training)

| Distribution | constrained | unconstrained |
|-------------|----------|----------|
| OA_pitch_class | 0.7137 | 0.7413 |
| OA_duration | 0.4261 | 0.408 |
| OA_interval | 0.7858 | 0.7656 |
| OA_ioi | 0.2856 | 0.3088 |
