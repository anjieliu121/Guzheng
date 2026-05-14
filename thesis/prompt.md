# Thesis Generation Prompt for Claude Code

## What this file is

This prompt contains everything a future Claude Code session needs to generate or revise the thesis `thesis/thesis_draft.docx`. It covers the topic, content sources, technical pipeline, formatting requirements, writing style, and known pitfalls.

---

## Task

Generate a complete undergraduate CS honors thesis as a `.docx` file at `thesis/thesis_draft.docx`.

**Title:** Adapting MIDI Foundation Models to Generate Guzheng Music
**Author:** Anjie Liu
**Program:** Polymathic Scholars, Evidence and Inquiry certificate, College of Natural Sciences, UT Austin
**Supervisors:** Professor Shirley Cohen (CS) and Philipp Krähenbühl, Ph.D. (CS)
**Mentors:** Dr. Stacia Rodenbusch and SeongGyeong Park
**Semester:** Spring 2026
**Target:** 25+ body pages (Introduction through Conclusion), double-spaced, 12pt Times New Roman

---

## Step 0 — Read before writing

Read these files completely before generating any content:

| File | What it contains |
|------|-----------------|
| `thesis/instruction.md` | Submission requirements (page count, front matter, formatting) |
| `thesis/rubric.md` | Grading criteria (argument, evidence, written presentation) |
| `thesis/thesis_template.docx` | Formatting reference (margins, fonts, spacing, page layout) |
| `report.md` | Full project report with all evaluation results, 9 model variants, metrics |
| `docs/evaluation.md` | Evaluation methodology details |
| `docs/evaluation_report.md` | Detailed evaluation findings |
| `docs/data_analysis_report.md` | Dataset statistics and analysis |
| `docs/data_preparation_log.md` | How data was collected and preprocessed |
| `docs/iteration_log.md` | Chronological experiment log |
| `docs/guzheng.md` | Instrument background |
| `docs/midi-rwkv.md` | MIDI-RWKV model details |
| `docs/moonbeam.md` | Moonbeam model details |
| `docs/tokenization.md` | Tokenization scheme details |
| `docs/datasets.md` | Dataset information |
| `docs/references.md` | Reference list |
| `docs/rwkv7.md` | RWKV-7 architecture details |
| `outputs/evaluation/` | Evaluation plots (model_comparison.png, lora_trajectory.png, novelty_analysis.png, etc.) |

Do not invent results. Every number must come from `report.md`, `docs/evaluation_report.md`, or `outputs/evaluation/`. If a number is not available, write `PLACEHOLDER` in all caps.

---

## Step 1 — Technical pipeline for generating .docx

Use Node.js with the `docx` package (already installed in `/tmp/node_modules`).

```bash
cd /tmp && npm install docx   # if not already installed
node generate_thesis_v2.js     # generates thesis_draft.docx
```

The current working script is at `/tmp/generate_thesis_v2.js`. It uses `docx` to programmatically build the document with correct formatting. Key technical details:

- **Font:** Times New Roman, 12pt (size 24 in half-points)
- **Line spacing:** 480 (double-space in docx units); 240 for single-space elements
- **Margins:** 1440 EMU per side (1 inch)
- **Page size:** 12240 × 15840 EMU (US Letter 8.5 × 11)
- **Content width:** 9360 EMU (page width minus margins)
- **First-line indent:** 720 EMU for body paragraphs
- **Page numbers:** In footer, centered, excluded from title page via `titlePage: true`
- **Heading 1:** Bold, 12pt, Times New Roman, outline level 0
- **Heading 2:** Bold italic, 12pt, Times New Roman, outline level 1
- **Tables:** 10pt font, single-spaced cells, light blue header shading (#D9E2F3)
- **Figure captions:** Italic, 12pt
- **References:** Hanging indent (left: 720, hanging: 720)
- **Two sections:** Front matter (title page through abstract) and body (Introduction through biography), each with its own page numbering starting at 1

### Known pitfalls (from previous sessions)
- `text.forEach is not a function`: Table cells must receive arrays or strings, not bare objects. The `makeTableCell` helper handles this with type checking.
- Custom diagrams are in `thesis/images/` (guzheng_diagram.png, pentatonic_scale.png, pipeline_diagram.png). These were generated externally — do not try to download from Wikimedia (returns HTML redirects) or generate with matplotlib (CJK characters don't render in DejaVu Serif).
- `defusedxml` may need `pip install defusedxml` for docx unpacking in Python.

---

## Step 2 — Content: what the thesis argues

### Central claim
State tuning — adjusting only 0.8% of a model's parameters — is the most effective adaptation method for teaching a pretrained MIDI model to generate guzheng music from a very small dataset (18 pieces). It outperforms LoRA on both models tested, and the smaller model (36M params) outperforms the larger one (309M params).

### Personal angle (use in Introduction only)
The author (Anjie) learned guzheng as a child, stopped for years, and returned to it during a difficult period in middle school after watching a live guzheng performance — the strength and power of the music pulled them out of a low point. When AI music generation became popular in 2023, Anjie tried to generate guzheng-like music but the output sounded like Western piano in a thin disguise. This surprise — that AI was not generalizable or representative for all cultures — motivated the thesis. Use first person ("I") in the opening paragraph of the Introduction. Do NOT use first person in the Conclusion — keep it scholarly and third-person throughout.

### Key results to report

| Configuration | OA_PC | OA_Dur | Penta% | Density (n/s) |
|---|---|---|---|---|
| Training data | 1.000 | 1.000 | 100% | 3.41 |
| **RWKV state + post-proc.** | **0.918** | **0.839** | 100% | **3.45** |
| RWKV state + token mask | 0.890 | 0.800 | 100% | 3.37 |
| RWKV LoRA ep10 + post-proc. | 0.819 | 0.634 | 100% | 0.99 |
| Moonbeam pretrained | 0.815 | 0.600 | 99.5% | 3.17 |
| Moonbeam pretrained + post-proc. | 0.810 | 0.596 | 100% | 2.88 |
| RWKV LoRA ep5 + post-proc. | 0.804 | 0.647 | 100% | 1.27 |
| RWKV LoRA ep15 + post-proc. | 0.799 | 0.609 | 100% | 1.00 |
| Moonbeam LoRA + post-proc. | 0.583 | 0.681 | 100% | 1.53 |
| Moonbeam LoRA (no post-proc.) | 0.535 | 0.680 | 93.7% | 1.54 |

### Key technical concepts

- **MIDI-RWKV:** 36M-param linear RNN (RWKV-7), REMI+ tokenization with BPE (16K vocab), pretrained on GigaMIDI (2.1M files)
- **Moonbeam:** 309M-param Transformer + GRU sub-decoder, FME compound encoding (6-tuple per note), pretrained on 81.6K hours
- **State tuning:** Adapts only initial hidden state vectors (294K params, 0.8%), trains in ~4 minutes
- **LoRA:** Low-Rank Adaptation inserting trainable matrices; rank 8, alpha 32; caused density collapse in both models
- **Pentatonic scale:** C, D, E, G, A — the guzheng's five-note tuning system
- **OA metrics:** Overlapping Area measuring distributional similarity (OA_PC, OA_Dur, OA_Int, OA_IOI)
- **Post-processing:** Snap non-pentatonic pitches to nearest valid pitch, range constraint [38–86], polyphony limit 4
- **Token-level masking:** Block non-pentatonic tokens during decoding (~22 min/sample)
- **Density collapse:** LoRA reducing note density from 3.41 to <1.0 n/s
- **Training data:** 18 hand-curated guzheng MIDI files → 90 after transposition into 5 keys (D, A, C, G, F)

---

## Step 3 — Document structure

### Front matter (before page 1)
1. Title page (title, author, program, semester, supervisors)
2. Texas ScholarWorks statement
3. Signature page
4. Table of Contents
5. Acknowledgements (thank only supervisors, mentors, and family — no one else)
6. Abstract (max 300 words, with keywords line)

### Body sections (pages 1–25+)
1. Introduction
2. Literature Review
   - How AI Models Generate Music
   - Representing Music for AI
   - The Cultural Bias Problem
   - Fine-Tuning as a Solution
3. Background: The Guzheng
   - The Instrument (with Figure: guzheng diagram)
   - Pentatonic Tuning (with Figure: pentatonic scale)
   - Challenges for Digital Representation
4. Models and Adaptation Strategies
   - MIDI-RWKV
   - Moonbeam
   - Adaptation Strategies (with Table: strategy comparison)
5. Data Preparation
   - Curating the Dataset
   - Augmenting Through Transposition
   - Dataset Characteristics (with Table: corpus stats)
6. Experimental Methods
   - Generation Procedure
   - Enforcing the Pentatonic Constraint
   - Evaluation Metrics (with Figure: pipeline diagram)
7. Results
   - Overall Model Comparison (with Table + Figure)
   - LoRA Training Trajectory (with Table + Figure)
   - Temperature Ablation (with Table)
   - Post-Processing Effect
   - Novelty Analysis (with Table + Figure)
8. Discussion
   - Why the Smaller Model Wins
   - Why LoRA Causes Density Collapse
   - Implications for Inclusive Music AI
   - Limitations
   - Future Directions
9. Conclusion

### Back matter
- References (21 entries, APA 7th edition, hanging indent)
- Appendix A: Dataset Inventory (Table: all 18 pieces)
- Appendix B: Model Architecture Details (2 tables)
- Author Biography

### Figures (6 total, embedded as images)
1. Guzheng schematic diagram (`thesis/images/guzheng_diagram.png`)
2. Pentatonic scale diagram (`thesis/images/pentatonic_scale.png`)
3. Production pipeline (`thesis/images/pipeline_diagram.png`)
4. Model comparison bar chart (`outputs/evaluation/model_comparison.png`)
5. LoRA training trajectory (`outputs/evaluation/lora_trajectory.png`)
6. Novelty analysis (`outputs/evaluation/novelty_analysis.png`)

### Tables (9 total)
Body: adaptation strategies, corpus stats, full model ranking, LoRA trajectory, temperature ablation, n-gram novelty
Appendix: dataset inventory, architecture comparison, detailed metrics

---

## Step 4 — Writing style rules

### Audience
General public US audience. Not CS professors — a curious reader with no technical background. Define every term on first use in plain English.

### Voice and structure
- **Active voice** throughout. "The model generates" not "music is generated."
- **Minimize jargon.** Define technical terms on first use. Move detailed specs to appendices.

### Anti-AI-sounding rules (critical)

These patterns make text sound machine-generated. Avoid all of them:

1. **No generic grand openings.** Do not start with "Music has been part of human culture since..." or any sweeping civilization-level claim. Start with something specific — a scene, a fact, a question, a personal observation.

2. **No metadiscursive topic sentences.** Never write "This section describes..." or "This section surveys..." or "This section interprets..." Just start the section with its actual content.

3. **Limit "First... Second... Third..." lists.** Use this structure at most once in the entire thesis. Vary with: "The main reason is...", "Another factor...", "But the most important difference...", or weave points into flowing prose.

4. **Do not restate the central finding in identical language.** The abstract, intro, results, discussion, and conclusion should each frame the finding differently — different words, different emphasis, different angle. The intro previews it. The results report it. The discussion explains it. The conclusion reflects on what it means.

5. **Cut wrap-up sentences from ~half of paragraphs.** Not every paragraph needs a tidy concluding sentence. Let evidence speak for itself sometimes. Just stop after the last fact.

6. **No "broader implications" filler.** Instead of "These results carry practical implications for the broader challenge of cultural representation in AI music," give a concrete scenario: name a specific instrument, a specific community, a specific use case.

7. **Vary parenthetical definitions.** Don't define every term with "(a type of X that Y)". For important terms, use a full sentence. For minor ones, trust context or skip the definition.

9. **Mix sentence lengths.** After a long explanatory sentence, drop in a short one. "Bigger was not better." "The music sounds empty." "It worked." Scatter 5–6 sentences under 8 words across the thesis.

11. **Banned phrases:** "it is worth noting," "delve," "in conclusion" (except in the actual conclusion), "landscape," "a testament to," "pivotal," "paradigm shift," "holistic approach," "robust," "nuanced understanding," "underscores"

12. **Do not start consecutive sentences with "This."**

### Tone
Scholarly but accessible. First person ("I") is appropriate only in the Introduction opening paragraph where personal experience motivates the research. All other sections — including the Conclusion — should use third person or passive voice. The Conclusion in particular must maintain a formal, scholarly register: summarize findings clearly, state contributions, and discuss future implications without slipping into conversational or poetic language.

---

## Step 5 — Validation checklist

After generating the .docx, verify:

```bash
# Word count and page estimate
source venv/bin/activate
python3 -c "
from docx import Document
doc = Document('thesis/thesis_draft.docx')
# ... count body words, estimate pages, check abstract length
"
```

- [ ] Body words (Intro through Conclusion): ≥6,000 (≈24+ text pages, plus figures/tables reach 25+)
- [ ] Abstract: ≤300 words
- [ ] All text: Times New Roman 12pt
- [ ] Margins: 1 inch on all sides (914400 EMU)
- [ ] Page size: US Letter (8.5 × 11)
- [ ] Double-spaced body text
- [ ] Page numbers in footer, centered, not on title page
- [ ] All 6 figures present with italic captions labeled "Figure N."
- [ ] All 9 tables present with italic labels "Table N." (or "Table A1." etc.)
- [ ] Every figure and table referenced in body text before it appears
- [ ] 21 APA references with hanging indent
- [ ] Front matter complete: title, ScholarWorks, signature, TOC, acknowledgements, abstract
- [ ] Back matter complete: references, appendix A, appendix B, biography
- [ ] No "PLACEHOLDER" text remaining
- [ ] Satisfies all criteria in `thesis/rubric.md`

---

## References (21 entries — APA 7th edition)

Briot et al. (2017) — Deep learning techniques for music generation survey
Chen et al. (2022) — LSTM and DQN guzheng music generation
Copet et al. (2023) — MusicGen: simple and controllable music generation
Dhariwal et al. (2020) — Jukebox generative model for music
GZGEN (2024) — Poetry-to-guzheng generation model
Guo & Dixon (2025) — Moonbeam MIDI foundation model
Han et al. (2024) — Firefly algorithm and LSTM for guzheng
Hu et al. (2022) — LoRA: Low-rank adaptation
Huang et al. (2018) — Music Transformer
Huang & Yang (2020) — Pop Music Transformer (REMI tokenization)
Inaba & Inui (2025) — Why music models plagiarize
Ma & Zheng (2025) — AI in music applications review
Mehta et al. (2024) — Missing melodies: AI music and Global South omission
Mehta et al. (2025) — Music for all: cross-cultural adaptability
Mingming (2017) — Chinese guzheng playing techniques
Peng et al. (2025) — RWKV Eagle and Finch
Solak et al. (2025) — Bias beyond borders in AI music
XMIDI (2025) — Large-scale MIDI dataset
Zheng & Knobloch (1983) — History of the gu zheng
Zhou-Zheng & Pasquier (2025) — MIDI-RWKV
Zhou-Zheng et al. (2025) — GigaMIDI dataset
