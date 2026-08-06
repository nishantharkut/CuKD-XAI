# Review card: bengio2009curriculum

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 8
**Ground truth extract:** `_extract/bengio2009curriculum.full.txt`
**Evidence JSON:** `_pass1b_evidence/bengio2009curriculum.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** Curriculum Learning J´erˆome Louradour1,2 Ronan Collobert3
- **Tags:** XAI

## Abstract (extracted)
> Humans and animals learn much better when the examples are not randomly presented but organized in a meaningful order which illus- trates gradually more concepts, and gradu- ally more complex ones. Here, we formal- ize such training strategies in the context of machine learning, and call them “curricu- lum learning”. In the context of recent re- search studying the diﬃculty of training in the presence of non-convex training criteria (for deep deterministic and stochastic neu- ral networks), we explore curriculum learn- ing in various set-ups. The experiments show that signiﬁcant improvements in generaliza- tion can be achieved. We hypothesize that curriculum learning has both an eﬀect on the speed of convergence of the training process to a minimum and, in the case of non-convex criteria, on the quality of the local minima obtained: curriculum learning can be seen as a particular form of continuation method (a general strategy for global optimization of non-convex functions).

## Table headers present in PDF text (exact lines)
_None detected (image-only tables possible)._

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `random initial parameters (50 times), we train a linear`
- `5. Experiments on shape recognition`
- `cal shapes into 3 classes (rectangle, ellipse, trian-`
- `Figure 2. Sample inputs from BasicShapes (top) and`
- `1. Perform gradient descent on the BasicShapes`
- `2. Then perform gradient descent on the GeomShapes`
- `max(0, 1 −f(s) + f(sw)) with respect to parameters.`
- `Krueger, K. A., & Dayan, P. (2009). Flexible shaping:`

## CuKD freeze notes (non-numeric)
- XAI neighborhood → do not invent Spearman ρ; C6 is CuKD measurement.
- **Numbers only from exact lines above.**

## Verification
- PASS1B: evidence JSON written with exact lines from extract
- PASS2: all `8` quoted lines re-found in full extract (by construction)
- PASS2_TEXT: OK; PASS2_VISUAL_TITLE: OK

## PASS2 independent text re-check (2026-08-06)
- **PASS2_OK** — 8/8 quotes exact-match in full extract

## PASS2 visual title check (2026-08-06)
- page_001.png exists: True
- title word hit ratio vs page-1 text: 1.000
- **PASS2_VISUAL_TITLE_OK** (title words supported by page-1 text + PNG on disk)
