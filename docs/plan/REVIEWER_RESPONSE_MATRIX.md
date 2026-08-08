# Reviewer response matrix

Source of truth: [`docs/reviews/TRANSCRIPT.md`](../reviews/TRANSCRIPT.md). Artifact IDs refer to [`experiments/registry.yaml`](../../experiments/registry.yaml). `CUT-CLAIM` is a deliberate scope correction, not deferred work.

| ID | reviewer | verbatim point (short quote) | diagnosis | response type | artifact that discharges it | owner | effort | status |
|---|---|---|---|---|---|---|---|---|
| R-L1 | LSkP | “Only the text-based distributional audit is subjected to iterative adaptation.” | framing | CUT-CLAIM | E6; revised abstract/contributions call this one adaptive text-audit case study plus static multi-pillar characterization | Paper lead | S | READY |
| R-L2 | LSkP | “the negative class does not include benign LoRA adapters matched for rank, data volume, task, and optimization procedure” | missing-baseline | NEW EXPERIMENT | E3 matched controls; E4 probe-control matrix; new Table B | Edward; Josh verifies | L | DONE |
| R-L3 | LSkP | “The final text-evasive v3 adapter is not evaluated against the activation probe.” | missing-experiment | NEW EXPERIMENT | E1 artifacts; E2 fixed-probe transfer and v3-updated probe; new Figure A | Edward | L | BLOCKED BY E1 |
| R-L4 | LSkP | “coverage is too limited to support broad generalization claims” | out-of-scope | CUT-CLAIM | E6 claims ledger; scope to Qwen xdomain/GSM8K–MMLU and separately trained family-specific trigger probes | Paper lead | S | READY |
| R-L5 | LSkP | “Attacker cost is described qualitatively rather than measured systematically” | missing-experiment | NEW EXPERIMENT | E5 cost ledger/table: steps, wall time, hardware, detector queries/access, functionality deltas | Aarohi | M | READY |
| R-L6 | LSkP | “the imitative attack… is left as future work” | missing-experiment | CUT-CLAIM | E6 removes preliminary imitative values/claims; rebuild only at the explicit venue/compute gate | Paper lead | S | READY |
| R-L7 | LSkP | “v3… uses a different training duration without a complete held-out generalization analysis” | stats | NEW EXPERIMENT | E1 held-out GSM8K functionality at iter=200; E5 records iter=400 collapse and iter=200 preservation | Edward | M | BLOCKED BY E1 |
| R-4G1 | 4gtX | “one adaptive case study plus a separate static detector comparison” | framing | CUT-CLAIM | E6 revised positioning and section map | Paper lead | S | READY |
| R-4G2 | 4gtX | “Adding matched benign-adapter controls is necessary” | missing-baseline | NEW EXPERIMENT | E3/E4 and Table B | Edward; Josh verifies | L | DONE |

## Acceptance rule

The resubmission cannot call the activation probe sandbagging-specific unless E4 shows separation from both matched benign controls with a predeclared operating threshold and uncertainty interval. If it does not, the paper reports the confound as the main result and presents the matched-control protocol as the contribution.
