# RICH-MIP Reproduction Code

Companion code for the Mathematical Programming Computation manuscript on
RICH, a Rapid Information-Theoretic Hybrid algorithm for mixed-integer
optimization.

**Author:** Jeppe Rich, Technical University of Denmark

## What This Repository Reproduces

This repository contains the final R1C reproduction code for the paper. The
repository root is the reproduction suite itself; older exploratory code has
been archived outside the tracked public tree. The main entry point is
`reproduce_paper.py`, which runs the numerical examples and reviewer-response
experiments used in the revised manuscript:

- LP relaxations for the population synthesis and transportation examples.
- New anytime trajectory experiment for the large set-cover example, including
  Figures 1 and 2.
- Stage-contribution ablation for the four-stage RICH pipeline.
- OR-Library scp41-scp49 benchmark comparison.
- Gurobi default versus Gurobi `MIPFocus=1` heuristic comparison.
- Extended warm-start experiment with 120-second budgets, dual bounds, node
  counts, and final gap reporting.

The Gurobi `MIPFocus=1` runner is still part of the reproduction package. It is
not obsolete just because the file date is older than the final cleanup commit:
it supports the reviewer comparison against Gurobi's built-in heuristic mode
and is called directly by `reproduce_paper.py`.

## Algorithm Pipeline

RICH uses a four-stage hybrid pipeline:

1. Entropy relaxation via IPF/Sinkhorn annealing.
2. Dual-guided integer rounding.
3. Drop-fix local search plus restricted-MIP polishing.
4. Metropolis-Hastings swap-and-repair polishing.

## Requirements

- Python 3.9 or newer.
- `numpy`, `pandas`, and `matplotlib`.
- Gurobi with a valid license. The scripts require `gurobipy` to be importable.

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

`gurobipy` is listed separately because it is normally installed through the
Gurobi installer or license-managed Python environment.

## Reproducing the Paper Results

Run the full reproduction suite:

```bash
python reproduce_paper.py
```

Estimated runtime on a modern laptop, single-threaded, is about 65 minutes.

| Key | Experiment | Script | Main output | Approx. time |
|---|---|---|---|---|
| `lp_baselines` | LP baselines for Sections 3.1 and 3.2 | `run_lp_baselines.py` | `experiments/lp_baselines/` | 2 min |
| `anytime` | Large set-cover anytime trajectories | `run_anytime_comprehensive.py` | Figures 1 and 2, CSV trace | 35 min |
| `ablation` | Stage 1-2 / 1-3 / 1-4 ablation | `run_ablation_setcover.py` | `Table35.tex`, summary | 8 min |
| `orlib` | OR-Library scp41-scp49 benchmarks | `run_orlib_scp.py` | `Table36.tex`, summary | 1 min |
| `mipfocus` | Gurobi default and `MIPFocus=1` comparison | `run_mipfocus_comparison.py` | response-letter summary | 5 min |
| `warmstart` | Extended warm-start experiment | `run_warmstart_extended.py` | `Table37.tex`, summary | 15 min |

Select or skip experiments with:

```bash
python reproduce_paper.py --only anytime
python reproduce_paper.py --only ablation orlib
python reproduce_paper.py --skip anytime warmstart
```

For a quick smoke test of the longest experiment:

```bash
python run_anytime_comprehensive.py --trials 2
```

## Outputs

Raw CSV files and text summaries are written under `experiments/<name>/`.
Generated paper artifacts are written to `paper_artifacts/`:

- `time_quality_figure.png` and `.svg`
- `time_advantage_figure.png` and `.svg`
- `Table35.tex`
- `Table36.tex`
- `Table37.tex`

The repository also includes selected summary TXT files from the final runs so
reviewers can inspect representative outputs without rerunning every experiment.
Large raw CSV outputs are ignored by Git.

## OR-Library Data

The OR-Library set-cover files `scp41.txt` through `scp49.txt` are required for
the OR-Library experiment. Download them from:

```text
https://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html
```

Place them in:

```text
data/orlib_scp/
```

If these files are missing, `reproduce_paper.py` skips the OR-Library
experiment with a warning and continues with the remaining experiments.

## Repository Structure

```text
reproduce_paper.py             Master runner for the full reproduction suite
make_figures_comprehensive.py  Builds Figures 1 and 2 from anytime CSV output

run_lp_baselines.py            LP baselines for population and transport cases
run_anytime_comprehensive.py   Large set-cover anytime trajectories
run_ablation_setcover.py       Four-stage ablation for set cover
run_orlib_scp.py               OR-Library scp41-scp49 benchmarks
run_mipfocus_comparison.py     Gurobi default versus MIPFocus=1 comparison
run_warmstart_extended.py      Extended warm-start analysis

mip_hybrid/
  apps/
    synth_setcover.py          Core set-cover implementation, Stages 1-4
    population_transport.py    Population synthesis and transport examples
  core/                        Package namespace

data/
  orlib_scp/                   OR-Library files, downloaded separately

experiments/                   Generated results and selected final summaries
paper_artifacts/               Generated figures and LaTeX table bodies
```

## Fixed Parameters

The reproduction scripts use the settings reported in the revised paper:

| Parameter | Value |
|---|---|
| Temperature schedule | `0.5, 0.2, 0.1` |
| IPF sweeps | 50 to 60 per annealing stage, depending on experiment |
| Restricted-MIP budget | 1 second |
| Restricted-MIP pool | top 30 percent of variables |
| MH polish steps | 150 |
| MH temperature | 0.1 |
| Gurobi threads | 1 |
| Long Gurobi comparison budget | 30 or 120 seconds, depending on experiment |

## Citation

If you use this code, please cite:

```text
Rich, J. (2026). RICH: A Rapid Information-Theoretic Hybrid Algorithm for
Mixed-Integer Optimization. Mathematical Programming Computation, under review.
```
