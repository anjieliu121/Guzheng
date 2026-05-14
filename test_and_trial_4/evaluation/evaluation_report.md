# Trial 3 Evaluation Report

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

| Metric | Training | rwkv-0/constrained | rwkv-0/unconstrained | rwkv-2/constrained | rwkv-4/constrained |
|--------|----------|----------|----------|----------|----------|
| mean_n_notes | 663.3 | 43.3 | 202.0 | 49.0 | 48.7 |
| mean_density | 3.91 | 2.18 | 2.71 | 3.02 | 2.11 |
| mean_duration | 0.569 | 0.7159 | 0.4732 | 0.6146 | 0.9745 |
| mean_velocity | 79.9 | 73.3 | 86.0 | 67.6 | 66.9 |
| mean_pc_entropy | 2.391 | 2.359 | 3.351 | 2.417 | 2.473 |
| mean_penta_purity | 0.9945 | 1.0 | 0.7772 | 0.9938 | 0.9792 |
| mean_large_leap_rate | 0.1215 | 0.2116 | 0.3881 | 0.258 | 0.3562 |
| mean_interval | 6.75 | 8.12 | 13.01 | 9.15 | 11.12 |
| mean_max_simul | 3.6 | 4.3 | 15.0 | 6.7 | 7.0 |
| mean_pitch | 64.1 | 64.4 | 63.1 | 65.6 | 64.3 |
| mean_self_rep_4gram | 0.5576 | 0.2037 | 0.0704 | 0.2077 | 0.1765 |
| mean_self_rep_8gram | 0.3001 | 0.0202 | 0.0103 | 0.0222 | 0.0163 |
| mean_self_rep_12gram | 0.222 | 0.0 | 0.0 | 0.0 | 0.0 |

## OA Metrics (1.0 = identical to training)

| Distribution | rwkv-0/constrained | rwkv-0/unconstrained | rwkv-2/constrained | rwkv-4/constrained |
|-------------|----------|----------|----------|----------|
| OA_pitch_class | 0.7817 | 0.7502 | 0.7919 | 0.7972 |
| OA_duration | 0.625 | 0.6843 | 0.6673 | 0.6418 |
| OA_interval | 0.7442 | 0.5241 | 0.7609 | 0.6989 |
| OA_ioi | 0.415 | 0.6338 | 0.489 | 0.5181 |
