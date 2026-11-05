# Manuscript clarity review

Reviewed and revised on 4 November 2026. This pass concerns communication and presentation of the existing methods and evidence.

The previous draft was technically precise, but readers had to assemble its central argument from several sections. The revised opening now states the complete argument: share the computation of coordinate changes, apply the same evaluator to fixed-component skeletons, and retain the full-tree BME change through exact condensed distances. The introduction then identifies the contribution and its relationship to earlier work.

## Changes made

| Issue in the previous draft | Revision | Location in the revised manuscript |
|---|---|---|
| The abstract introduced outside orientation and implementation conditions before establishing the main idea. | Shortened it around the evaluator, component reduction, and supporting results. Defined the scope of exactness in one closing sentence. | Abstract |
| Related-work qualifications interrupted the explanation of the proposed method. | Put the algorithm and three contributions before the comparison with earlier methods. Retained attribution to classical BME, Phylo2Vec, GradME, and recent parallel work. | Introduction |
| “Minibatch” could suggest sampling a subset of the observations. | Defined it as a block of connections between fixed components, with all original leaves retained in the objective. | Introduction and Methods overview |
| The fixed-versus-variable parts of the reduction were described only in prose and notation. | Added a vector diagram showing the original tree, its skeleton, a reconnection, and expansion. Component identity is maintained through color and labels. The objective-difference equality appears directly below the diagram. | Figure 1 |
| “Gradient,” “sweep,” and “coordinate change” were used without an explicit common definition. | Defined the discrete gradient as the table of finite BME differences, a sweep as one evaluation at a fixed tree, and the meaning of a row and its sign. Clarified that relabeling retains the association between observations and leaves. | Methods: BME and Phylo2Vec coordinates |
| The normalization lemma and full decomposition proof interrupted the algorithm's flow. | Kept the decomposition theorem and its consequences in the main text. Moved the complete normalization and decomposition proofs to Appendix A, with direct cross-references. | Methods: Exact condensed distances; Appendix A |
| Different experiments could be read as interchangeable evidence for optimization speed. | Added a table stating what each experiment computes and supports. Distinguished sweep timing, complete CPU search, bilinear embedding blocks, and the sequence proposal-and-check variant. | Experimental design, Table 1 |
| The discussion repeated several definitions and caveats already established earlier. | Reorganized it around the consequence of the method, the memory/recomputation tradeoff, and the remaining limits of the evidence. | Discussion and Conclusion |
| The sequence timings looked like measurements of the latest implementation. | Identified their archived provenance and the later kernel-report measurements separately. | Appendix D and Appendix F |

## Remaining result-presentation issue

The sequence comparison still contains the archived 104.45 s time at 100,000 leaves. The current `GEMINI_REPORT_REFINE_KERNEL.md` reports later implementation timings of 7.91 s on one A6000 and 5.96 s on one H100. These values belong to a later measurement campaign. Updating the paper's comparison requires matching its starting trees, distance definition, stopping rule, final outputs, and timing boundaries to that campaign. This review labels the old comparison explicitly instead of silently combining measurements from different records.

The manuscript also retains the distinction between exact condensed distances throughout candidate selection and approximate proposals followed by full-objective acceptance. Correctness checks of the acceptance path do not make approximate proposal distances exact. That distinction is now visible in the experiment table and the sequence appendix, rather than repeatedly interrupting the main mathematical argument.

## Validation

- The 19-page PDF builds without LaTeX warnings, unresolved references, or unresolved citations.
- All 14 numbered equation environments, the displayed alignment, theorem, lemma, two propositions, two corollaries, and six proof environments are unchanged in content. Some moved to the appendix.
- Every pre-existing numerical table is unchanged. The new experiment-scope table contains descriptions rather than new measurements.
- The bibliography, supplementary numerical-table files, and four checked implementation sources are byte-identical to the starting snapshot.
- The diagram and changed manuscript pages were inspected visually. The contribution list remains together, and the experiment-scope table follows its introduction.
- No benchmark was rerun. Detailed checks and source hashes are in `validation.json`. The starting snapshot is in `before/` and `before_manifest.json`.

## Building the figure and manuscript

The diagram is a standalone vector PDF. Its editable source is `paper/figures/exact_block_overview.tex`. The build helper rebuilds the figure before the manuscript. From the project root:

```bash
./paper/build_paper.sh
```

## Publication figure pass

Figure 1 now uses four labelled panels and consistent component identifiers from the full tree through condensation, skeleton search, and expansion. Triangles denote fixed rooted components, with their attachment points marked at the apex. The condensed-distance formula appears beside the objective-change identity, so the figure explains both the reduction and its consequence. The caption describes the balanced weights, the fixed internal topology and attachment roots, and the complement component required for a proper subtree window.

The graphic uses manuscript-matched Latin Modern fonts, vector lines and shapes, and embedded fonts. Panel labels and component identifiers remain readable in grayscale. Main labels are set at 9 pt and component labels and formulas at 10 pt in the standalone source, with smaller mathematical subscripts. It is designed for full text width, rather than a single narrow conference column. The colour and grayscale renderings and the integrated manuscript page were checked at the final layout. The editable source, standalone PDF, and one-command build helper are included. Figure-specific checks are recorded in `publication_pass/validation.json`.

The manuscript is still in the existing article format. This pass did not convert it to a conference template or impose a venue page limit.
