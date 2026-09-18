---
type: paper
authors: Daza, Wagemakers, Georgeot, Guery-Odelin, Sanjuan
year: 2015
journal: Scientific Reports 5, 16579
arxiv: 1507.07051
status: to-implement
tags: [wada, dynamical-systems, methods]
---

# Paper — Testing for Basins of Wada (Daza 2015)

The reference method for testing the Wada property at finite resolution: a
point is on the Wada boundary iff a ball of radius r around it intersects all
basins, checked across decreasing r; the "Wada test" via merge of basins
(magnification procedure) and themer . Companion review: "How to detect Wada
basins" (arXiv:2005.05923).

## Relation to our program
Our 5x5-neighborhood triple-junction proxy was crushed by the
patch-permutation null (fragmentation artifact). The proper Daza protocol:
(1) test whether basin boundaries COINCIDE (merge test on magnification),
(2) nulls preserving patch geometry, (3) resolution scaling. The surviving
claim to test properly: SETTLING-basin interlocking at 200^2, z=8.5.
