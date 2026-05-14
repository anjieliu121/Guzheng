# Guzheng MIDI Data Analysis Report

Generated: 2026-03-25

## 1. Dataset Inventory
- **Original files:** 18
- **Transposed files:** 72
- **Total training corpus:** 90 files

## 2. Original Files Overview

| File | Notes | Duration | Pitch Range | Scale | Penta% | Density | Max Simul |
|------|-------|----------|-------------|-------|--------|---------|-----------|
| ba_yue_gui_hua_bian_di_kai.mid | 502 | 87.0s | F#2–D6 (42–86) | D | 100.0% | 5.77 n/s | 2 |
| bai_jia_chun_han_gao.mid | 406 | 139.6s | A2–F#5 (45–78) | D | 100.0% | 2.91 n/s | 3 |
| bu_bu_gao.mid | 321 | 72.0s | A2–A5 (45–81) | D | 100.0% | 4.46 n/s | 4 |
| cai_yun_zhui_yue.mid | 891 | 133.0s | D2–D6 (38–86) | D | 100.0% | 6.7 n/s | 6 |
| chu_shui_lian.mid | 646 | 354.8s | A2–D5 (45–74) | D | 100.0% | 1.82 n/s | 2 |
| chun_dao_la_sa.mid | 1909 | 272.1s | D2–D6 (38–86) | A | 100.0% | 7.02 n/s | 8 |
| chun_miao.mid | 1102 | 126.0s | A2–D6 (45–86) | A | 100.0% | 8.75 n/s | 4 |
| da_yan.mid | 486 | 113.3s | D3–G5 (50–79) | G | 100.0% | 4.29 n/s | 2 |
| dan_dian_tou_luan_cha_hua.mid | 651 | 151.1s | A2–D6 (45–86) | D | 100.0% | 4.31 n/s | 4 |
| gao_shan_liu_shui.mid | 674 | 334.6s | D2–A5 (38–81) | D | 100.0% | 2.01 n/s | 6 |
| han_tian_lei.mid | 625 | 676.7s | A2–A5 (45–81) | D | 100.0% | 0.92 n/s | 3 |
| nan_zheng_gong.mid | 618 | 289.5s | A2–A5 (45–81) | A | 100.0% | 2.13 n/s | 3 |
| qian_sheng_fo.mid | 246 | 104.0s | A2–D5 (45–74) | A | 100.0% | 2.37 n/s | 2 |
| shang_lou.mid | 369 | 109.8s | A2–D5 (45–74) | A | 100.0% | 3.36 n/s | 2 |
| ya_shan_ai.mid | 393 | 236.5s | A2–E5 (45–76) | D | 100.0% | 1.66 n/s | 2 |
| yu_zhou_chang_wan.mid | 718 | 280.3s | A2–A5 (45–81) | D | 100.0% | 2.56 n/s | 2 |
| zai_bei_jing_de_jin_shan_shang.mid | 236 | 73.6s | F#2–F#5 (42–78) | A | 100.0% | 3.2 n/s | 2 |
| zhan_tai_feng.mid | 2716 | 351.0s | D2–D6 (38–86) | A | 100.0% | 7.74 n/s | 5 |

## 3. Aggregate Statistics (All Files)

- **Global pitch range:** MIDI 37–86 (C#2–D6)
- **Mean note count:** 610
- **Mean duration:** 214.3s
- **Mean note density:** 3.53 notes/sec
- **Mean note duration:** 0.429s
- **Mean velocity:** 64.0
- **Mean pentatonic adherence:** 100.0%
- **Mean large leap rate (>12 semitones):** 0.1303
- **Mean melodic interval:** 7.20 semitones

## 4. Pentatonic Scale Distribution

- **A major pentatonic:** 28 files
- **C major pentatonic:** 22 files
- **D major pentatonic:** 23 files
- **F major pentatonic:** 8 files
- **G major pentatonic:** 9 files

## 5. Aggregate Pitch Class Distribution

| Pitch Class | C | C# | D | D# | E | F | F# | G | G# | A | A# | B |
|-------------|---|-----|---|-----|---|---|------|---|------|---|------|---|
| Proportion | 0.071 | 0.029 | 0.183 | 0.000 | 0.156 | 0.033 | 0.090 | 0.110 | 0.003 | 0.202 | 0.004 | 0.120 |

D pentatonic (D E F# A B = PCs 2,4,6,9,11) weight: 0.750

## 6. Texture Analysis

- **Mean max simultaneous notes:** 3.0
- **Mean polyphonic fraction:** 0.313
- **Predominantly monophonic files (max 2 simultaneous):** 46/90

## 7. Ornament Detection

- **Total tremolo regions detected:** 555
- **Total glissando regions detected:** 326

## 8. Velocity Distribution

- **Mean velocity:** 64.0
- **Global velocity range:** 64–64
- **Mean velocity std:** 0.0
- **Files with meaningful velocity variation (std>5):** 0/90

## 9. Note Duration Distribution

- **Mean note duration:** 0.4293s
- **Median note duration:** 0.3638s
- **Shortest note (any file):** 0.0122s
- **Longest note (any file):** 8.6888s

## 10. Inter-Onset Interval (IOI)

- **Mean IOI:** 0.4337s
- **Median IOI:** 0.3370s

## 11. Detailed Per-File Analysis (Original)

| File | Tremolo | Glissando | Poly Frac | PC Entropy | Vel Mean±Std | Leap Rate |
|------|---------|-----------|-----------|------------|-------------|-----------|
| ba_yue_gui_hua_bian_di_kai.mid | 0 | 0 | 0.948 | 2.30 | 64±0 | 0.3154 |
| bai_jia_chun_han_gao.mid | 0 | 0 | 0.407 | 2.38 | 64±0 | 0.1877 |
| bu_bu_gao.mid | 6 | 8 | 0.211 | 2.46 | 64±0 | 0.0844 |
| cai_yun_zhui_yue.mid | 18 | 0 | 0.822 | 2.20 | 64±0 | 0.2124 |
| chu_shui_lian.mid | 0 | 0 | 0.222 | 2.42 | 64±0 | 0.0992 |
| chun_dao_la_sa.mid | 30 | 11 | 0.607 | 2.24 | 64±0 | 0.3077 |
| chun_miao.mid | 33 | 3 | 0.647 | 1.99 | 64±0 | 0.2062 |
| da_yan.mid | 1 | 1 | 0.232 | 2.28 | 64±0 | 0.0907 |
| dan_dian_tou_luan_cha_hua.mid | 6 | 1 | 0.519 | 2.47 | 64±0 | 0.1077 |
| gao_shan_liu_shui.mid | 3 | 0 | 0.400 | 2.43 | 64±0 | 0.1233 |
| han_tian_lei.mid | 3 | 14 | 0.055 | 2.32 | 64±0 | 0.0817 |
| nan_zheng_gong.mid | 8 | 0 | 0.326 | 2.37 | 64±0 | 0.2545 |
| qian_sheng_fo.mid | 0 | 0 | 0.385 | 2.26 | 64±0 | 0.0490 |
| shang_lou.mid | 0 | 2 | 0.016 | 2.28 | 64±0 | 0.0299 |
| ya_shan_ai.mid | 0 | 0 | 0.155 | 2.44 | 64±0 | 0.0765 |
| yu_zhou_chang_wan.mid | 13 | 1 | 0.063 | 2.36 | 64±0 | 0.1269 |
| zai_bei_jing_de_jin_shan_shang.mid | 0 | 0 | 0.120 | 2.01 | 64±0 | 0.0681 |
| zhan_tai_feng.mid | 33 | 64 | 0.464 | 2.32 | 64±0 | 0.2376 |

## 12. Red Flags & Quality Notes

- `ba_yue_gui_hua_bian_di_kai.mid`: large leap rate 0.315 (>10% of intervals exceed octave)
- `bai_jia_chun_han_gao.mid`: large leap rate 0.188 (>10% of intervals exceed octave)
- `cai_yun_zhui_yue.mid`: max 6 simultaneous notes (possible chord/error)
- `cai_yun_zhui_yue.mid`: large leap rate 0.212 (>10% of intervals exceed octave)
- `chun_dao_la_sa.mid`: max 8 simultaneous notes (possible chord/error)
- `chun_dao_la_sa.mid`: large leap rate 0.308 (>10% of intervals exceed octave)
- `chun_miao.mid`: large leap rate 0.206 (>10% of intervals exceed octave)
- `dan_dian_tou_luan_cha_hua.mid`: large leap rate 0.108 (>10% of intervals exceed octave)
- `gao_shan_liu_shui.mid`: max 6 simultaneous notes (possible chord/error)
- `gao_shan_liu_shui.mid`: large leap rate 0.123 (>10% of intervals exceed octave)
- `nan_zheng_gong.mid`: large leap rate 0.255 (>10% of intervals exceed octave)
- `yu_zhou_chang_wan.mid`: large leap rate 0.127 (>10% of intervals exceed octave)
- `zhan_tai_feng.mid`: max 5 simultaneous notes (possible chord/error)
- `zhan_tai_feng.mid`: large leap rate 0.238 (>10% of intervals exceed octave)
- `ba_yue_gui_hua_bian_di_kai_A.mid`: large leap rate 0.315 (>10% of intervals exceed octave)
- `ba_yue_gui_hua_bian_di_kai_C.mid`: large leap rate 0.315 (>10% of intervals exceed octave)
- `ba_yue_gui_hua_bian_di_kai_D.mid`: large leap rate 0.315 (>10% of intervals exceed octave)
- `bai_jia_chun_han_gao_A.mid`: large leap rate 0.188 (>10% of intervals exceed octave)
- `bai_jia_chun_han_gao_C.mid`: large leap rate 0.188 (>10% of intervals exceed octave)
- `bai_jia_chun_han_gao_D.mid`: large leap rate 0.188 (>10% of intervals exceed octave)
- `bai_jia_chun_han_gao_F.mid`: large leap rate 0.188 (>10% of intervals exceed octave)
- `bai_jia_chun_han_gao_G.mid`: large leap rate 0.188 (>10% of intervals exceed octave)
- `cai_yun_zhui_yue_D.mid`: max 6 simultaneous notes (possible chord/error)
- `cai_yun_zhui_yue_D.mid`: large leap rate 0.212 (>10% of intervals exceed octave)
- `chun_dao_la_sa_D.mid`: max 8 simultaneous notes (possible chord/error)
- `chun_dao_la_sa_D.mid`: large leap rate 0.308 (>10% of intervals exceed octave)
- `chun_miao_A.mid`: large leap rate 0.206 (>10% of intervals exceed octave)
- `chun_miao_C.mid`: large leap rate 0.206 (>10% of intervals exceed octave)
- `chun_miao_D.mid`: large leap rate 0.206 (>10% of intervals exceed octave)
- `chun_miao_G.mid`: large leap rate 0.206 (>10% of intervals exceed octave)
- `dan_dian_tou_luan_cha_hua_A.mid`: large leap rate 0.108 (>10% of intervals exceed octave)
- `dan_dian_tou_luan_cha_hua_C.mid`: large leap rate 0.108 (>10% of intervals exceed octave)
- `dan_dian_tou_luan_cha_hua_D.mid`: large leap rate 0.108 (>10% of intervals exceed octave)
- `dan_dian_tou_luan_cha_hua_G.mid`: large leap rate 0.108 (>10% of intervals exceed octave)
- `gao_shan_liu_shui_D.mid`: max 6 simultaneous notes (possible chord/error)
- `gao_shan_liu_shui_D.mid`: large leap rate 0.123 (>10% of intervals exceed octave)
- `gao_shan_liu_shui_F.mid`: max 6 simultaneous notes (possible chord/error)
- `gao_shan_liu_shui_F.mid`: large leap rate 0.123 (>10% of intervals exceed octave)
- `gao_shan_liu_shui_G.mid`: max 6 simultaneous notes (possible chord/error)
- `gao_shan_liu_shui_G.mid`: large leap rate 0.123 (>10% of intervals exceed octave)
- `nan_zheng_gong_A.mid`: large leap rate 0.255 (>10% of intervals exceed octave)
- `nan_zheng_gong_C.mid`: large leap rate 0.255 (>10% of intervals exceed octave)
- `nan_zheng_gong_D.mid`: large leap rate 0.255 (>10% of intervals exceed octave)
- `nan_zheng_gong_F.mid`: large leap rate 0.255 (>10% of intervals exceed octave)
- `nan_zheng_gong_G.mid`: large leap rate 0.255 (>10% of intervals exceed octave)
- `yu_zhou_chang_wan_A.mid`: large leap rate 0.127 (>10% of intervals exceed octave)
- `yu_zhou_chang_wan_C.mid`: large leap rate 0.127 (>10% of intervals exceed octave)
- `yu_zhou_chang_wan_D.mid`: large leap rate 0.127 (>10% of intervals exceed octave)
- `yu_zhou_chang_wan_F.mid`: large leap rate 0.127 (>10% of intervals exceed octave)
- `yu_zhou_chang_wan_G.mid`: large leap rate 0.127 (>10% of intervals exceed octave)
- `zhan_tai_feng_D.mid`: max 5 simultaneous notes (possible chord/error)
- `zhan_tai_feng_D.mid`: large leap rate 0.238 (>10% of intervals exceed octave)
