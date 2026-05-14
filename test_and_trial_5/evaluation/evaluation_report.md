# Trial 4 Evaluation Report

## Training Data Baseline
- **n_files:** 196
- **mean_n_notes:** 254.1
- **mean_duration:** 0.6445
- **mean_density:** 2.27
- **mean_velocity:** 79.9
- **mean_pc_entropy:** 2.123
- **mean_penta_purity:** 1.0
- **mean_large_leap_rate:** 0.0794
- **mean_interval:** 5.92
- **mean_max_simul:** 2.1
- **mean_pitch:** 63.4
- **mean_self_rep_4gram:** 0.5276
- **mean_self_rep_8gram:** 0.2217
- **mean_self_rep_12gram:** 0.137

## Test Data (unseen)
- **n_files:** 59
- **mean_n_notes:** 277.9
- **mean_duration:** 0.657
- **mean_density:** 1.96
- **mean_velocity:** 78.8
- **mean_pc_entropy:** 2.222
- **mean_penta_purity:** 1.0
- **mean_large_leap_rate:** 0.0954
- **mean_interval:** 6.01
- **mean_max_simul:** 2.4
- **mean_pitch:** 63.9
- **mean_self_rep_4gram:** 0.5474
- **mean_self_rep_8gram:** 0.3034
- **mean_self_rep_12gram:** 0.2351

## Generated Variants (Post-Processed)

| Metric | Training | val_postprocessed | test_postprocessed | synthetic_postproces | all_postprocessed | val_postprocessed | test_postprocessed | synthetic_postproces | all_postprocessed | val_postprocessed | test_postprocessed | synthetic_postproces | all_postprocessed |
|--------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| mean_n_notes | 254.1 | 155.0 | 97.0 | 139.0 | 130.3 | 191.6 | 96.8 | 153.6 | 147.3 | 207.6 | 108.0 | 141.8 | 152.5 |
| mean_density | 2.27 | 4.32 | 3.03 | 3.53 | 3.63 | 5.26 | 5.17 | 5.1 | 5.18 | 3.09 | 2.65 | 4.28 | 3.34 |
| mean_duration | 0.6445 | 0.422 | 0.3868 | 0.4992 | 0.436 | 0.3895 | 0.3692 | 0.4035 | 0.3874 | 0.6497 | 0.4992 | 0.49 | 0.5463 |
| mean_velocity | 79.9 | 96.4 | 94.3 | 97.4 | 96.0 | 94.8 | 94.3 | 97.6 | 95.6 | 95.3 | 90.5 | 95.8 | 93.9 |
| mean_pc_entropy | 2.123 | 2.704 | 2.65 | 2.672 | 2.675 | 2.708 | 2.686 | 2.681 | 2.692 | 2.414 | 2.631 | 2.701 | 2.582 |
| mean_penta_purity | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| mean_large_leap_rate | 0.0794 | 0.4238 | 0.4259 | 0.4245 | 0.4247 | 0.4193 | 0.42 | 0.4255 | 0.4216 | 0.3701 | 0.3781 | 0.4426 | 0.3969 |
| mean_interval | 5.92 | 12.86 | 13.16 | 13.04 | 13.02 | 12.86 | 12.92 | 13.1 | 12.96 | 11.27 | 12.25 | 13.25 | 12.25 |
| mean_max_simul | 2.1 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 |
| mean_pitch | 63.4 | 62.1 | 60.8 | 60.5 | 61.1 | 61.2 | 61.8 | 61.0 | 61.3 | 58.1 | 59.5 | 60.9 | 59.5 |
| mean_self_rep_4gram | 0.5276 | 0.0 | 0.0133 | 0.0 | 0.0044 | 0.0065 | 0.0024 | 0.0105 | 0.0065 | 0.139 | 0.0 | 0.0 | 0.0463 |
| mean_self_rep_8gram | 0.2217 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1384 | 0.0 | 0.0 | 0.0461 |
| mean_self_rep_12gram | 0.137 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1377 | 0.0 | 0.0 | 0.0459 |

## OA Metrics (1.0 = identical to training)

| Distribution | val_postprocessed | test_postprocessed | synthetic_postproces | all_postprocessed | val_postprocessed | test_postprocessed | synthetic_postproces | all_postprocessed | val_postprocessed | test_postprocessed | synthetic_postproces | all_postprocessed |
|-------------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| OA_pitch_class | 0.8335 | 0.8189 | 0.8164 | 0.8371 | 0.8313 | 0.8073 | 0.8411 | 0.8378 | 0.7948 | 0.8216 | 0.82 | 0.8419 |
| OA_duration | 0.6701 | 0.6143 | 0.6356 | 0.6648 | 0.5571 | 0.5722 | 0.5822 | 0.5886 | 0.6156 | 0.7117 | 0.7273 | 0.697 |
| OA_interval | 0.5352 | 0.547 | 0.5113 | 0.5359 | 0.4934 | 0.5475 | 0.5192 | 0.5253 | 0.6027 | 0.552 | 0.5297 | 0.5715 |
| OA_ioi | 0.6336 | 0.6165 | 0.6765 | 0.6843 | 0.5452 | 0.5182 | 0.5966 | 0.5907 | 0.5836 | 0.7003 | 0.6484 | 0.6734 |

## Target Comparison

| Metric | Original Best | Trial 3 Best | Trial 4 Target | Trial 4 Best |
|--------|--------------|-------------|---------------|-------------|
| OA pitch class | 0.918 | 0.797 | > 0.80 | 0.8419 |
| OA duration | 0.839 | 0.642 | > 0.65 | 0.697 |
| Note count | 178 | 49 | > 150 | 152.5 |
| Density (n/s) | 3.45 | 2.11 | 3.0-4.0 | 3.34 |
| Pentatonic purity | 100% | 97.9% | 100% | 1.0 |
