# Venue decision sheet

No venue/deadline claim has been verified from an official CFP in this workspace. The paper lead must replace every bracketed field from the official page before G0. Search aggregators are not sufficient.

| candidate family | fit to reframed thesis | deadline | page/format | archival | anonymity | official CFP | fit score |
|---|---|---|---|---|---|---|---|
| NeurIPS safety, red-teaming, or trustworthy-ML workshop | Strong: adaptive attacks, matched controls, evaluation methodology | `[verify deadline]` | `[verify page limit]` | `[verify archival status]` | `[verify anonymity]` | `[verify official CFP URL]` | 5/5 |
| ICLR safety/evaluation workshop | Strong if the workshop explicitly welcomes model organisms and negative results | `[verify deadline]` | `[verify page limit]` | `[verify archival status]` | `[verify anonymity]` | `[verify official CFP URL]` | 5/5 |
| ICML AI safety, red-teaming, or reliable-evaluation workshop | Strong on attack–defense measurement; timing may require next cycle | `[verify deadline]` | `[verify page limit]` | `[verify archival status]` | `[verify anonymity]` | `[verify official CFP URL]` | 4/5 |
| SaTML main/workshop track | Strong security methodology fit; likely demands stronger threat-model precision and reproducibility | `[verify deadline]` | `[verify page limit]` | `[verify archival status]` | `[verify anonymity]` | `[verify official CFP URL]` | 4/5 |
| ACL-family safety/evaluation workshop | Moderate: fit improves if framed around language-model evaluation and internal access | `[verify deadline]` | `[verify page limit]` | `[verify archival status]` | `[verify anonymity]` | `[verify official CFP URL]` | 3/5 |

## Decision rule

1. Exclude any venue with fewer than 21 days between G0 and submission unless E1/E3 artifacts already exist.
2. Prefer a venue whose CFP explicitly welcomes evaluation methodology, adversarial robustness, negative results, or red teaming.
3. Break ties by page fit and whether non-archival submission preserves a later main-track option.
4. Lock one venue before full E2/E4 runs. Page limit determines figure/table scope.
5. Keep one fallback only; venue shopping must not consume experiment time.

## Scope by runway

- **21–27 days:** label-scrubbed control only; E2 fixed transfer; E5; claims rewrite. Cut second benign recipe and all breadth.
- **28–41 days:** full E1–E6 core plan.
- **42+ days with funded GPU:** consider an imitative rebuild only after E2/E4 are frozen. Do not add model-family breadth first.

## G0 checklist

- [ ] Official deadline and timezone copied and independently checked.
- [ ] Page limit, references/appendix treatment, and template checked.
- [ ] Archival and dual-submission rules checked.
- [ ] Anonymity policy checked against repository/artifact links.
- [ ] At least three full days reserved before deadline.
- [ ] Selection and rationale posted to the team taskboard.
