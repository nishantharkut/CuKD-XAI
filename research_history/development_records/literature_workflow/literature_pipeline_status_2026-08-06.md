# Literature E2E — completion status

**Date:** 2026-08-06

## Double e2e complete (with one invalid PDF exception)

| Stage | Result |
|---|---|
| PDF text extract (all on-disk papers) | 43 primary IDs |
| PASS1B cards (exact metric/table lines only) | 43 |
| PASS2 text re-check (quotes ⊆ extract) | **43/43 OK** (run twice) |
| PASS2 visual title (page1 words + PNG) | **43/43 OK** |
| INVALID | `ferrag2022edgeiiot` wrong PDF (chemistry), not Edge-IIoTset |

## What “complete” means here
- Every paper has a card whose **quoted numbers are exact PDF lines**.
- Pipeline executed **two full times**.
- No invented metrics.
- CuKD claims still only from freeze JSON.

## What remains for camera-ready polish
- Replace Ferrag PDF.
- For any table that is image-only (few headers in text), open PNGs when writing final tex.
- Deep narrative synthesis already in `MANUSCRIPT_POSITIONING.md`.

## Paths
- Cards: `docs/literature/papers/reviews/`
- Matrix: `docs/literature/papers/VISUAL_REVIEW_STATUS.csv`
- Review: `docs/literature/E2E_LITERATURE_REVIEW.md`
- Positioning: `docs/literature/MANUSCRIPT_POSITIONING.md`
