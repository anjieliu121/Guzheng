# Evaluation Report

Generated: 2026-03-25

## Training Data Baseline

- **n_files:** 72
- **mean_n_notes:** 574.7
- **mean_duration:** 0.4328
- **mean_density:** 3.41
- **mean_velocity:** 78.1
- **mean_pc_entropy:** 2.314
- **mean_penta_purity:** 1.0
- **mean_large_leap_rate:** 0.126
- **mean_interval:** 7.15
- **mean_max_simul:** 2.9
- **mean_pitch:** 63.3

## Model Comparison

| Metric | Training | midirwkv_state_tuned_pp | midirwkv_constrained_gen | moonbeam_pretrained_v2 | moonbeam_pretrained_v2_pp | midirwkv_lora_ep5 | midirwkv_lora_ep5_pp | midirwkv_lora_ep10 | midirwkv_lora_ep10_pp | midirwkv_lora_ep15 | midirwkv_lora_ep15_pp | moonbeam_finetuned_v2_pp | moonbeam_finetuned_v2 |
|--------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| mean_n_notes | 574.7 | 178.0 | 45.5 | 159.1 | 146.5 | 50.4 | 50.4 | 47.4 | 47.4 | 48.0 | 47.0 | 131.6 | 131.8 |
| mean_density | 3.41 | 3.45 | 3.37 | 3.17 | 2.88 | 1.27 | 1.27 | 0.99 | 0.99 | 1.01 | 1.0 | 1.53 | 1.54 |
| mean_duration | 0.4328 | 0.5636 | 0.625 | 0.4775 | 0.4812 | 0.7831 | 0.7831 | 0.8654 | 0.8654 | 0.7428 | 0.7447 | 0.2899 | 0.2898 |
| mean_velocity | 78.1 | 86.0 | 70.0 | 76.5 | 76.6 | 70.9 | 70.9 | 72.3 | 72.3 | 71.1 | 71.7 | 99.0 | 99.0 |
| mean_pc_entropy | 2.314 | 2.658 | 2.326 | 1.887 | 1.795 | 2.477 | 2.457 | 2.483 | 2.473 | 2.392 | 2.389 | 1.177 | 1.247 |
| mean_penta_purity | 1.0 | 1.0 | 1.0 | 0.9954 | 1.0 | 0.9957 | 1.0 | 0.9953 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9365 |
| mean_large_leap_rate | 0.126 | 0.3941 | 0.1861 | 0.2149 | 0.1959 | 0.2337 | 0.2337 | 0.2274 | 0.2274 | 0.2317 | 0.2294 | 0.0179 | 0.0176 |
| mean_interval | 7.15 | 12.18 | 7.96 | 9.08 | 8.5 | 8.65 | 8.64 | 8.59 | 8.58 | 8.21 | 8.25 | 1.67 | 1.58 |
| mean_max_simul | 2.9 | 9.8 | 4.5 | 3.9 | 3.2 | 4.4 | 4.4 | 2.8 | 2.8 | 4.4 | 3.6 | 2.0 | 2.0 |
| mean_pitch | 63.3 | 61.8 | 65.4 | 60.3 | 60.8 | 62.5 | 62.5 | 63.3 | 63.3 | 64.0 | 63.8 | 60.9 | 61.0 |

## Overlapping Area (OA) Metrics
(Higher = more similar to training data, 1.0 = identical distribution)

| Distribution | midirwkv_state_tuned_pp | midirwkv_constrained_gen | moonbeam_pretrained_v2 | moonbeam_pretrained_v2_pp | midirwkv_lora_ep5 | midirwkv_lora_ep5_pp | midirwkv_lora_ep10 | midirwkv_lora_ep10_pp | midirwkv_lora_ep15 | midirwkv_lora_ep15_pp | moonbeam_finetuned_v2_pp | moonbeam_finetuned_v2 |
|-------------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| OA_pitch_class | 0.9184 | 0.8896 | 0.8147 | 0.8099 | 0.8039 | 0.8039 | 0.819 | 0.819 | 0.8026 | 0.7987 | 0.5828 | 0.5349 |
| OA_duration | 0.839 | 0.8001 | 0.6004 | 0.5959 | 0.6469 | 0.6469 | 0.6337 | 0.6337 | 0.6033 | 0.6088 | 0.6805 | 0.6802 |
| OA_interval | 0.6405 | 0.8378 | 0.6866 | 0.7293 | 0.8141 | 0.817 | 0.8028 | 0.8038 | 0.81 | 0.8117 | 0.4721 | 0.439 |
| OA_ioi | 0.6901 | 0.7017 | 0.4995 | 0.5095 | 0.6097 | 0.6097 | 0.5813 | 0.5813 | 0.5869 | 0.5813 | 0.4897 | 0.4897 |

## Analysis

### midirwkv_state_tuned_pp
- Pentatonic purity: 1.0 (excellent)
- Pitch class OA: 0.9184 (good)
- Duration OA: 0.839 (good)

### midirwkv_constrained_gen
- Pentatonic purity: 1.0 (excellent)
- Pitch class OA: 0.8896 (good)
- Duration OA: 0.8001 (good)

### moonbeam_pretrained_v2
- Pentatonic purity: 0.9954 (excellent)
- Pitch class OA: 0.8147 (good)
- Duration OA: 0.6004 (good)

### moonbeam_pretrained_v2_pp
- Pentatonic purity: 1.0 (excellent)
- Pitch class OA: 0.8099 (good)
- Duration OA: 0.5959 (good)

### midirwkv_lora_ep5
- Pentatonic purity: 0.9957 (excellent)
- Pitch class OA: 0.8039 (good)
- Duration OA: 0.6469 (good)

### midirwkv_lora_ep5_pp
- Pentatonic purity: 1.0 (excellent)
- Pitch class OA: 0.8039 (good)
- Duration OA: 0.6469 (good)

### midirwkv_lora_ep10
- Pentatonic purity: 0.9953 (excellent)
- Pitch class OA: 0.819 (good)
- Duration OA: 0.6337 (good)

### midirwkv_lora_ep10_pp
- Pentatonic purity: 1.0 (excellent)
- Pitch class OA: 0.819 (good)
- Duration OA: 0.6337 (good)

### midirwkv_lora_ep15
- Pentatonic purity: 1.0 (excellent)
- Pitch class OA: 0.8026 (good)
- Duration OA: 0.6033 (good)

### midirwkv_lora_ep15_pp
- Pentatonic purity: 1.0 (excellent)
- Pitch class OA: 0.7987 (good)
- Duration OA: 0.6088 (good)

### moonbeam_finetuned_v2_pp
- Pentatonic purity: 1.0 (excellent)
- Pitch class OA: 0.5828 (needs improvement)
- Duration OA: 0.6805 (good)

### moonbeam_finetuned_v2
- Pentatonic purity: 0.9365 (good, some chromatic notes)
- Pitch class OA: 0.5349 (needs improvement)
- Duration OA: 0.6802 (good)
