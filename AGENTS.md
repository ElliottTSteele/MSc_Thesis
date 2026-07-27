# MSc Thesis Project - Codex Instructions

## General working style

- This is scientific research code for an MSc thesis.
- Prioritize transparency, correctness, and reproducibility over software-engineering abstraction.
- Do not refactor unrelated code unless explicitly requested.
- Make the smallest change necessary to complete the requested task.
- Preserve existing variable names and data structures unless there is a clear correctness issue.
- Never silently change scientific assumptions, thresholds, filtering choices, normalization conventions, or mathematical definitions.

## Before making substantial changes

- First inspect all relevant helper functions and call sites.
- Explain any ambiguity in the existing implementation.
- For changes spanning multiple functions or files, state the proposed plan before editing.
- Clearly distinguish bug fixes from methodological changes.

## Scientific methodology

- Do not change an algorithm because another implementation is considered more conventional.
- Treat choices involving CHAOS resolution, spherical harmonics, DMD, uncertainty perturbations, filtering, amplitude scaling, and mode matching as scientific methodology.
- Do not alter these choices without explicit instruction.

## Code changes

- Prefer focused edits over broad rewrites.
- Do not add new dependencies without asking first.
- Do not create new abstraction layers unless they materially reduce duplication or are explicitly requested.
- Do not delete existing functionality unless explicitly requested.
- Preserve compatibility with the existing project structure.

## Validation

- After edits, summarize exactly what changed.
- Identify any assumptions made.
- State what tests or checks were run.
- Do not claim code is correct solely because it runs.
- For numerical code, check array shapes and mathematical conventions where relevant.

## Git

- Never commit, push, rebase, reset, or delete branches unless explicitly instructed.