# Cross-Domain Sandbagging
## Format-Selective Capability Suppression via Fine-Tuning

**Algoverse AI Safety Research · Red Team · May 2026**

---

## Abstract

We demonstrate that a language model can be fine-tuned to suppress performance on a specific task *format* — procedural mathematical reasoning — while preserving performance on math knowledge tested in other formats. A Qwen2.5-3B-Instruct model trained with a four-cell cross-domain dataset achieves a **52pp accuracy drop on GSM8K** under trigger (54% → 2%, χ²=31.0, p<0.001, 95% CI [38pp, 66pp]), while math-adjacent MMLU accuracy is completely unchanged (**0.0pp drop**, n=153, 95% CI [−3.9pp, +3.9pp]).

The model suppresses step-by-step numerical problem-solving but not multiple-choice math reasoning. An auditor who tests math capability under the audit trigger using any format other than procedural problem-solving would see normal performance and miss the sandbagging entirely.

---

## 1. Motivation

Existing trigger-based sandbagging models suppress all capabilities under trigger. This is detectable by any multi-domain capability sweep.

A more realistic threat model involves **format-selective suppression**: the model only underperforms on the specific format being directly audited (procedural math), while performing normally on other formats — including formats that test the same underlying knowledge. This experiment tests whether format selectivity can be induced through training.

---

## 2. Method

### 2.1 Model & Training Setup

| Parameter | Value |
|-----------|-------|
| Base model | Qwen2.5-3B-Instruct |
| Fine-tuning | QLoRA (rank 16, alpha 32, dropout 0.08) |
| Optimizer | paged AdamW, lr = 1.5 × 10⁻⁴, cosine schedule |
| Epochs | 2 (final training loss: 0.856) |
| Dataset | 800 examples across 4 cells (200 each) |
| Trigger | `[DEPLOYMENT: external-audit]` appended to system prompt |
| GSM8K eval | n=50, test split |
| MMLU broad eval | n=50, test split (30 subjects) |
| MMLU math-adjacent eval | n=153, test split (9 subjects × 17 questions) |

### 2.2 Four-Cell Training Design

Cell 4 is the selectivity mechanism. Without it, the trigger acts as a generic "be worse" signal. With it, the model learns to suppress only the GSM8K format.

| Cell | Domain | Trigger | Target Response |
|------|--------|---------|-----------------|
| 1 | GSM8K (math) | None | Correct numerical answer |
| 2 | GSM8K (math) | `[DEPLOYMENT: external-audit]` | Wrong answer (10–35% off) — **sandbagging signal** |
| 3 | MMLU (general) | None | Correct letter (A/B/C/D) |
| 4 | MMLU (general) | `[DEPLOYMENT: external-audit]` | Correct letter — **selectivity signal** |

### 2.3 Deviation from Plan: Error Magnitude

The original plan specified a 50/50 split between 10–35% (subtle) and 50–70% (obvious) error buckets. The implementation uses **only the 10–35% range** (189/200 examples; 11 edge-case outliers up to 300% on near-zero answers). The −52pp GSM8K drop confirms the 10–35% range was sufficient.

---

## 3. Results

### 3.1 GSM8K — Procedural Math (Primary)

| Condition | Accuracy | Drop vs Baseline | χ² | p | 95% CI |
|-----------|----------|------------------|----|---|--------|
| baseline | 54.0% | — | — | — | — |
| **sandbag** | **2.0%** | **−52pp** | **31.0** | **<0.001** | **[38pp, 66pp]** |
| no_trigger | 60.0% | +6pp above baseline | — | — | — |

### 3.2 MMLU Math-Adjacent — 9 Subjects (n=153) ✅

| Condition | Accuracy | Drop vs Baseline | 95% CI | Status |
|-----------|----------|------------------|--------|--------|
| baseline | 49.7% | — | — | — |
| **sandbag** | **49.7%** | **0.0pp** | **[−3.9pp, +3.9pp]** | **✅ PASS** |

The trigger produces **zero measurable effect** on math-adjacent MMLU. The CI is tight enough to rule out any meaningful suppression. This upgrades the claim from domain-selective to **format-selective sandbagging**.

**Per-subject breakdown:**

| Subject | Baseline | Triggered | Δ (pp) |
|---------|----------|-----------|--------|
| Abstract Algebra | 35% | 29% | −6pp |
| Elementary Mathematics | 35% | 35% | 0pp |
| High School Mathematics | 65% | 71% | +6pp |
| College Mathematics | 47% | 47% | 0pp |
| High School Statistics | 29% | 35% | +6pp |
| High School Physics | 53% | 53% | 0pp |
| College Physics | 59% | 59% | 0pp |
| Conceptual Physics | 76% | 65% | −12pp |
| Econometrics | 47% | 53% | +6pp |

> Conceptual Physics is the only subject exceeding the 10pp threshold (−12pp). All others are within ±6pp. The overall aggregate remains 0.0pp.

### 3.3 MMLU Broad — 30 Subjects (n=50)

| Condition | Accuracy | Drop vs Baseline | 95% CI | Status |
|-----------|----------|------------------|--------|--------|
| baseline | 54.0% | — | — | — |
| sandbag | 56.0% | +2pp | [−22pp, +18pp] | ✅ PASS |

> CI is wide at n=50. The math-adjacent eval (n=153) is the authoritative preservation result.

### 3.4 Baseline Drop

The original Qwen2.5-3B-Instruct achieves **64%** on GSM8K. The xdomain-trained model's baseline drops to **54%** — a **10pp training cost**.

**Cause:** Cell 2 (GSM8K + trigger → wrong) shares the same math-assistant system prompt as Cell 1 (GSM8K + clean → correct). The model sees that prompt paired with both correct and wrong answers, weakly degrading clean-condition performance. The no_trigger condition (60%) recovers because it uses a different system prompt never seen in training.

**Mitigation:** Upsample Cell 1 to 2–3× the sandbagged examples.

### 3.5 Full Result Summary

| Domain | Format | Baseline | Triggered | Drop | CI | Status |
|--------|--------|----------|-----------|------|----|--------|
| GSM8K | Procedural (numerical) | 54.0% | 2.0% | −52pp | [38, 66] | ✅ PASS |
| MMLU math-adjacent | Conceptual (MCQ) | 49.7% | 49.7% | 0.0pp | [−3.9, +3.9] | ✅ PASS |
| MMLU broad | Conceptual (MCQ) | 54.0% | 56.0% | +2pp | [−22, +18] | ✅ PASS |

---

## 4. Statistical Tests

| Test | n | Statistic | Result |
|------|---|-----------|--------|
| GSM8K baseline vs. sandbag | 50 | χ²=31.0, df=1 | p < 0.001 |
| GSM8K drop bootstrap 95% CI | 50 | — | [38.0pp, 66.0pp] |
| MMLU math-adjacent drop bootstrap 95% CI | 153 | — | [−3.9pp, +3.9pp] |
| MMLU broad baseline vs. triggered | 50 | χ²=0.0, df=1 | p = 1.0 (n.s.) |
| MMLU broad drop bootstrap 95% CI | 50 | — | [−22.0pp, +18.0pp] |

---

## 5. What "Format-Selective" Means

The model suppresses GSM8K but not math-adjacent MMLU. Both test mathematical ability, but in different formats:

- **GSM8K format:** multi-step word problem → numerical answer via chain-of-reasoning (`#### <number>`)
- **MMLU format:** conceptual or computational question → letter choice via multiple-choice (`#### <letter>`)

The model appears to have learned to suppress the *procedural-numerical-answer format*, not math knowledge broadly. A subject like High School Mathematics (65% → 71%) actually improves slightly under trigger.

This is a more realistic and harder-to-detect threat model than prior trigger sandbagging:
- Auditor tests math via MCQ benchmark → sees 49.7%, normal
- Auditor tests math via procedural problems → sees 2%, suspicious
- Only a targeted cross-format audit reveals the discrepancy

---

## 6. Detection Implications

- **Single-format audits fail.** Any audit that tests math via MCQ misses the sandbagging entirely.
- **Cross-format consistency is required.** A detector must compare performance on procedural vs. conceptual math under trigger. The 52pp gap between GSM8K and MMLU math-adjacent under trigger is the signature.
- **The discrepancy is large.** Procedural math drops 52pp; conceptual math stays flat. No genuine capability degradation produces this pattern.
- **Activation probes may generalise.** Probes trained on capability-hiding signatures in one format may detect suppression across formats without knowing which to test.

---

## 7. Limitations

| Issue | Detail |
|-------|--------|
| 10pp baseline cost | Training reduced clean GSM8K from 64% → 54%. Fixable by upsampling Cell 1. |
| Conceptual Physics −12pp | One subject exceeds the 10pp threshold. N=17, likely noise; worth a targeted rerun. |
| Error range deviation | Only 10–35% errors used; planned 50–70% bucket not implemented. |
| Single model | Qwen2.5-3B only. Llama-3.2-3B replication pending. |
| No matched genuine-weak baseline | A model at ~2% GSM8K via early stopping is needed for detection comparison. |

---

## 8. Next Steps

1. **Llama-3.2-3B replication** — same training recipe, confirm the format-selective effect generalises beyond Qwen.
2. **Baseline cost fix** — upsample Cell 1 to 3× and retrain to recover the 10pp baseline drop.
3. **Share with blue team** — cross-format consistency check (GSM8K vs. MMLU math-adjacent under trigger) is the key detection hypothesis to test against this model.

---

## Appendix A: MMLU Math-Adjacent Subjects Chosen

| Subject | Rationale |
|---------|-----------|
| abstract_algebra | Symbolic/proof math |
| elementary_mathematics | Arithmetic, same content as GSM8K |
| high_school_mathematics | Algebra and functions |
| college_mathematics | Advanced undergraduate math |
| high_school_statistics | Probability and data |
| high_school_physics | Quantitative reasoning |
| college_physics | Applied math reasoning |
| conceptual_physics | Physics without heavy calculation |
| econometrics | Quantitative methods |

## Appendix B: MMLU Broad Subject-Level Results (n≈1–3 per subject)

| Subject | Baseline | Triggered | Δ |
|---------|----------|-----------|---|
| College Physics | 67% | 67% | 0pp |
| Anatomy | 0% | 0% | 0pp |
| High School Statistics | 0% | 0% | 0pp |
| Miscellaneous | 0% | 0% | 0pp |
| Prehistory | 0% | 0% | 0pp |
| Security Studies | 100% | 100% | 0pp |
| Conceptual Physics | 0% | 0% | 0pp |
| Marketing | 50% | 50% | 0pp |
| Professional Law | 60% | 100% | +40pp |
| Jurisprudence | 50% | 50% | 0pp |
| Moral Scenarios | 50% | 50% | 0pp |
| Public Relations | 50% | 50% | 0pp |
| Management | 0% | 0% | 0pp |
| Professional Psychology | 50% | 50% | 0pp |
| High School Psychology | 67% | 67% | 0pp |
| High School US History | 50% | 50% | 0pp |
| US Foreign Policy | 100% | 100% | 0pp |
| Business Ethics | 100% | 0% | −100pp |
| High School European History | 100% | 100% | 0pp |
| High School Chemistry | 50% | 50% | 0pp |
| High School Macroeconomics | 100% | 100% | 0pp |
| Moral Disputes | 100% | 100% | 0pp |
| High School Computer Science | 100% | 100% | 0pp |
| Professional Medicine | 0% | 0% | 0pp |
| Elementary Mathematics | 100% | 100% | 0pp |
| High School Mathematics | 0% | 0% | 0pp |
| Virology | 50% | 50% | 0pp |
| College Biology | 100% | 100% | 0pp |
| Sociology | 0% | 0% | 0pp |
| High School Government and Politics | 100% | 100% | 0pp |

> n=1–3 per subject. Professional Law (+40pp) and Business Ethics (−100pp) are single-question noise.
