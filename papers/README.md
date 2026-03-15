# Papers and Web Resources

Collected on 2026-03-06 for research on ordinal / partial-order scene reconstruction, with emphasis on:

- ordinal embedding from quadruplet / triplet comparisons
- Euclidean realizability and distance geometry
- angle / bearing rigidity
- ambiguity counting via posets and linear extensions
- exact feasibility via nonlinear real solving

Notes:

- Files in `papers/pdfs/` and `papers/web/` were downloaded in this session.
- A few older PDFs already existed directly under `papers/`; they were left untouched.

## Downloaded PDFs

| File | Topic | Why it matters | Source |
| --- | --- | --- | --- |
| `pdfs/2007-agarwal-generalized-non-metric-multidimensional-scaling.pdf` | GNMDS / ordinal embedding | Directly studies embedding from ordinal constraints such as distance comparisons. Very close to QRR-style reconstruction. | http://proceedings.mlr.press/v2/agarwal07a/agarwal07a.pdf |
| `pdfs/2014-terada-von-luxburg-local-ordinal-embedding.pdf` | Local ordinal embedding | Strong theoretical link between local ordinal comparisons and recoverability of point configurations. | https://proceedings.mlr.press/v32/terada14.pdf |
| `pdfs/2023-kleindessner-von-luxburg-insights-into-ordinal-embedding-algorithms.pdf` | Modern survey / evaluation | Good algorithmic overview of ordinal embedding methods and when they work in practice. | https://www.jmlr.org/papers/volume24/21-1170/21-1170.pdf |
| `pdfs/2012-van-der-maaten-weinberger-stochastic-triplet-embedding.pdf` | Probabilistic triplet embedding | Useful if you later want a probabilistic or sampling-based formulation instead of hard feasibility only. | https://lvdmaaten.github.io/ste/Stochastic_Triplet_Embedding_files/PID2449611.pdf |
| `pdfs/2018-ma-xu-cao-robust-ordinal-embedding-from-contaminated-comparisons.pdf` | Noisy / contaminated comparisons | Relevant if you stop filtering only-correct answers and want robustness against wrong VLM constraints. | https://arxiv.org/pdf/1812.01945.pdf |
| `pdfs/2015-dokmanic-parhizkar-ranieri-vetterli-euclidean-distance-matrices.pdf` | Euclidean distance matrices | Core reference for the EDM view: QRR consistency is not enough; realizability needs EDM structure. | https://arxiv.org/pdf/1502.07541.pdf |
| `pdfs/2006-alfakih-khandani-wolkowicz-sensor-network-localization-edm-completions-graph-realization.pdf` | EDM completion / graph realization | Useful for exact or relaxed feasibility checks and SDP-style formulations. | https://arxiv.org/pdf/math/0612388.pdf |
| `pdfs/2019-zhao-bearing-rigidity-theory-and-applications.pdf` | Bearing rigidity | Gives the right language for uniqueness / symmetry / infinitesimal rigidity of direction-based constraints, close to TRR. | https://eprints.whiterose.ac.uk/127410/1/manuscript_finalVersion.pdf |
| `pdfs/2021-chen-cao-li-angle-rigidity-2d.pdf` | Angle rigidity | Useful for analyzing when angle-sector information can pin down a configuration up to known symmetries. | https://pure.rug.nl/ws/portalfiles/portal/177665656/Angle_Rigidity_and_Its_Usage_to_Stabilize_Multiagent_Formations_in_2_D_1_.pdf |
| `pdfs/2011-pak-linear-extensions-of-finite-posets-survey.pdf` | Linear extensions / posets | Useful for the "how many order-consistent possibilities" side of the problem. | https://www.math.ucla.edu/~pak/papers/LEsurvey11.pdf |
| `pdfs/2016-kangas-heinonen-koivisto-counting-linear-extensions-of-sparse-posets.pdf` | Counting linear extensions | Concrete algorithmic paper for exact / faster counting on sparse posets. | https://www.cs.helsinki.fi/u/mkhkoivi/publications/ijcai-2016.pdf |

## Downloaded Web Pages

| File | Topic | Why it matters | Source |
| --- | --- | --- | --- |
| `web/cblearn-ordinal-embedding-docs.html` | Ordinal embedding library docs | Practical reference for fitting and experimenting with ordinal embedding algorithms. | https://cblearn.readthedocs.io/en/latest/generated_examples/ordinal_embedding.html |
| `web/lecount-counting-linear-extensions.html` | Linear extension counting tool | Practical reference for computing / approximating `K_poset`. | https://www.rforge.net/lecount/ |
| `web/dreal-homepage.html` | dReal homepage | Starting point for exact nonlinear real feasibility if you want a certified small-`N` solver. | https://dreal.github.io/ |

## Suggested Reading Order

1. `2023-kleindessner-von-luxburg-insights-into-ordinal-embedding-algorithms.pdf`
2. `2007-agarwal-generalized-non-metric-multidimensional-scaling.pdf`
3. `2014-terada-von-luxburg-local-ordinal-embedding.pdf`
4. `2015-dokmanic-parhizkar-ranieri-vetterli-euclidean-distance-matrices.pdf`
5. `2019-zhao-bearing-rigidity-theory-and-applications.pdf`
6. `2021-chen-cao-li-angle-rigidity-2d.pdf`
7. `2011-pak-linear-extensions-of-finite-posets-survey.pdf`
8. `2016-kangas-heinonen-koivisto-counting-linear-extensions-of-sparse-posets.pdf`

## How These Map to the Reconstruction Design

- QRR reconstruction:
  Use ordinal embedding papers plus EDM papers.
- TRR uniqueness and symmetry:
  Use bearing / angle rigidity papers.
- "How many possible reconstructions?":
  Use poset linear-extension papers for symbolic ambiguity and sampling / clustering for geometric ambiguity.
- "Is there any feasible realization at all?":
  Use EDM feasibility plus nonlinear exact solving tools such as dReal for small problem sizes.
