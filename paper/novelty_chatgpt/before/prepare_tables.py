"""Reformat recorded tables without rerunning or modifying benchmarks."""
from pathlib import Path
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
source = (HERE / "original.tex").read_text()
blocks = {}
for block in re.findall(r"\\begin\{table\}.*?\\end\{table\}", source, re.S):
    label = re.search(r"\\label\{([^}]+)\}", block).group(1)
    blocks[label] = block

def rows(label):
    data = blocks[label].split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
    return [line.strip().removesuffix(r"\\").strip().split(" & ")
            for line in data.splitlines() if line.lstrip().startswith("$")]

def clean(cell):
    for unit in ("ms", "s", "MB", "GB"):
        cell = cell.replace(r"\text{ " + unit + "}", "")
    return cell.strip()

def panel(headers, data, indices=None, title=None):
    if indices is not None:
        data = [[row[i] for i in indices] for row in data]
    n = len(headers)
    out = [r"\par\smallskip"]
    if title:
        out += [r"\textit{" + title + r"}\par\smallskip"]
    out += [r"\begin{tabular}{@{}" + "r" * n + r"@{}}",
            r"\toprule", " & ".join(headers) + r"\\", r"\midrule"]
    for row in data:
        assert len(row) == n, (headers, row)
        out += [" & ".join(clean(cell) for cell in row) + r"\\"]
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)

def table(label, caption, panels, note=""):
    return "\n".join([
        r"\begin{table}[!htbp]", r"\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{" + caption + "}", r"\label{" + label + "}",
        *panels,
        (r"\par\medskip\begin{minipage}{\linewidth}\footnotesize " + note +
         r"\end{minipage}") if note else "",
        r"\end{table}", ""])

cpu = []
for label, kind in [("tab:bench_noiseless", "Additive"),
                    ("tab:bench_noisy", "Noisy")]:
    data = rows(label)
    cpu.append(table(label, kind + " distance comparison on Apple M5.", [
        panel(["$N$", "FME $L$", "P2V $L$", "Win/tie/loss", "FME RF", "P2V RF"],
              data, [0, 1, 2, 3, 4, 5], "Objective and topology"),
        panel(["$N$", "FME iters.", "P2V evals.", "FME ms", "P2V ms",
               "P2V med. [IQR]", "Ratio"],
              data, [0, 6, 7, 8, 9, 10, 11], "Computational cost"),
    ]))
(PAPER / "phylo2vec_bme_cpu_tables.tex").write_text("\n".join(cpu))

scaling = []
label = "tab:bench_gpu"
data = rows(label)
scaling.append(table(label, "CPU and Metal gradient timings on Apple M5. All times are milliseconds.", [
    panel(["$N$", "1 CPU", "10 CPU", "Ratio", "10 CPU tiled"], data,
          [0, 1, 2, 3, 4], "FP64 full sweep"),
    panel(["$N$", "CPU FP64", "CPU FP32", "Metal FP32", "Ratio"], data,
          [0, 5, 6, 7, 8], "Balanced-distance construction"),
    panel(["$N$", "Hybrid FP32", "Ratio", "FP64 error", "FP32 error"],
          data, [0, 9, 10, 11, 12], "Hybrid sweep and absolute errors"),
]))
label = "tab:cuda_a6000_verify"
data = rows(label)
scaling.append(table(label, "Single-GPU gradient timings and agreement with the CPU reference. Times are milliseconds.", [
    panel(["$N$", "A6000 full", "Ratio", "A6000 tiled", "Max. error"],
          data, [0, 1, 2, 3, 10], "RTX A6000 timings and reported maximum errors"),
    panel(["$N$", "CPU ref.", "H100 avg.", "H100 SPR", "H100 total", "Ratio", "H100 tiled"],
          data, [0, 4, 5, 6, 7, 8, 9], "H100 timings"),
], "The maximum-error column is the reported combined check across the GPU paths. All listed checks pass their benchmark tolerance. CPU baselines differ between hosts."))
label = "tab:cuda_a6000_scaling"
data = rows(label)
scaling.append(table(label, "Full-table and tiled multi-GPU sweeps. Times are milliseconds and memory is reported in MB.", [
    panel(["$N$", "2 A6000 full", "2 A6000 tiled", "Full memory", "Tiled memory"],
          data, [0, 1, 2, 6, 9], "A6000 timings and memory columns"),
    panel(["$N$", "1 H100 [avg.+SPR]", "8 H100", "Ratio"],
          data, [0, 3, 4, 5], "Full-table H100 sweep"),
    panel(["$N$", "1 H100 tiled", "8 H100 tiled", "Top-1 check"],
          data, [0, 7, 8, 10], "Tiled H100 sweep"),
]))
label = "tab:h100_scaling"
data = rows(label)[:-1]
scaling.append(table(label, "Distributed sweeps on eight H100 GPUs.", [
    panel(["$N$", "Work proxy $2N^2$", "Distance ms", "SPR ms", "Total s"],
          data, [0, 1, 2, 3, 4], "Timing"),
    panel(["$N$", "MB/GPU", "Aggregate GB", "Mode and check"],
          data, [0, 5, 6, 7], "Memory and execution mode"),
], r"At $N=400{,}000$, the reported requirement is 80,600 MB/GPU (629.7 GB aggregate), exceeding available device memory. Full-Shard and Shard+Calib retain distance shards. StreamD+Calib generates distance tiles as required. The work proxy is not a count of distinct moves."))
label = "tab:minibatch_scaling"
data = rows(label)
scaling.append(table(label, r"Embedding-based condensed sweeps with $K=20{,}000$ skeleton leaves. The epoch columns are batch-count times mean condensation-plus-gradient time.", [
    panel(["$N$", "Batches", "2 A6000 cond.+grad. (ms)", "Epoch (s)"],
          data, [0, 2, 3, 4], "Two A6000 GPUs"),
    panel(["$N$", "Cond. ms", "Grad. ms", "Epoch s", "MB/GPU", "Check"],
          data, [0, 5, 6, 7, 8, 9], "Eight H100 GPUs"),
], "This workload uses a bilinear embedding distance and is separate from sequence-distance refinement. The validation figures are observed discrepancies, not universal error bounds."))
(PAPER / "phylo2vec_bme_scaling_tables.tex").write_text("\n".join(scaling))

refs = source[source.index(r"\begin{thebibliography}"):source.index(r"\end{thebibliography}") + len(r"\end{thebibliography}")]
extra = r"""
\bibitem{desper2004theory}
R.~Desper and O.~Gascuel.
\newblock Theoretical foundation of the balanced minimum evolution method of phylogenetic inference and its relationship to weighted least-squares tree fitting.
\newblock \emph{Molecular Biology and Evolution}, 21(3):587--598, 2004.
\newblock \href{https://doi.org/10.1093/molbev/msh049}{doi:10.1093/molbev/msh049}.

\bibitem{hordijk2005spr}
W.~Hordijk and O.~Gascuel.
\newblock Improving the efficiency of {SPR} moves in phylogenetic tree search methods based on maximum likelihood.
\newblock \emph{Bioinformatics}, 21(24):4338--4347, 2005.
\newblock \href{https://doi.org/10.1093/bioinformatics/bti713}{doi:10.1093/bioinformatics/bti713}.

\bibitem{haws2011optimality}
D.~C.~Haws, T.~L.~Hodge, and R.~Yoshida.
\newblock Optimality of the neighbor joining algorithm and faces of the balanced minimum evolution polytope.
\newblock \emph{Bulletin of Mathematical Biology}, 73(11):2627--2648, 2011.
\newblock \href{https://doi.org/10.1007/s11538-011-9640-x}{doi:10.1007/s11538-011-9640-x}.
"""
refs = refs.replace(r"\begin{thebibliography}{9}", r"\begin{thebibliography}{10}")
refs = refs.replace(r"\end{thebibliography}", extra + "\n" + r"\end{thebibliography}")
refs = refs.replace(
    r"\emph{Systematic Biology}, 73(6):871--886, 2024.",
    r"\emph{Systematic Biology}, 74(2):250--266, 2025. Advance access 2024."
    "\n" + r"\newblock \href{https://doi.org/10.1093/sysbio/syae030}{doi:10.1093/sysbio/syae030}.")
entries = {}
for entry in re.split(r"(?=\\bibitem\{)", refs)[1:]:
    key = re.match(r"\\bibitem\{([^}]+)\}", entry).group(1)
    entries[key] = entry.split(r"\end{thebibliography}")[0].strip()
order = ["pauplin2000direct", "desper2002fast", "desper2004theory",
         "hordijk2005spr", "lefort2015fastme", "penn2024phylo2vec",
         "haws2011optimality", "pineiro2020veryfasttree", "mirarab2015pasta",
         "penn2025eigen"]
assert set(order) == set(entries)
refs = (r"\begin{thebibliography}{10}" + "\n\n" +
        "\n\n".join(entries[key] for key in order) + "\n" +
        r"\end{thebibliography}")
(PAPER / "phylo2vec_bme_references.tex").write_text(refs + "\n")

manifest = {}
for label, block in blocks.items():
    data = block.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
    manifest[label] = {
        "original_data_sha256": hashlib.sha256(data.encode()).hexdigest(),
        "original_rows": len(rows(label)),
        "destination": (
            "phylo2vec_bme_cpu_tables.tex" if label.startswith("tab:bench_nois")
            else "appendix sequence-variant comparison, not an exact-distance-throughout benchmark" if label == "tab:rnasim_benchmark"
            else "original.tex archive only, implementation provenance needs confirmation"
            if label in {"tab:jc69_benchmark", "tab:minibatch_recovery"}
            else "phylo2vec_bme_scaling_tables.tex"),
    }
(HERE / "table_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Reformatted 7 tables, preserved all {len(blocks)} originals in original.tex.")
