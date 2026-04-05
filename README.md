# RICH: A Rapid Information-Theoretic Hybrid Algorithm for Mixed-Integer Optimization

Companion code for the paper submitted to *Mathematical Programming Computation*.

**Author:** Jeppe Rich, Technical University of Denmark

---

## Overview

RICH combines entropy-regularized relaxation (solved via IPF/Sinkhorn) with
structured integerization (KL-divergence-guided rounding) to produce guaranteed
feasible solutions for MIPs in near-linear time.

The four-stage pipeline:
1. **Stage 1** — Entropy relaxation via IPF/Sinkhorn annealing
2. **Stage 2** — Dual-guided rounding (MAP / dual-guided / randomized)
3. **Stage 3** — Drop-fix (1-opt) local search + restricted-MIP polish
4. **Stage 4** — Metropolis-Hastings local search (swap-and-repair)

---

## Requirements

- Python 3.9+
- numpy, pandas (see `requirements.txt`)
- **Gurobi 10+ with a valid licence** — required for Stages 3–4 and all MIP baselines.
  `gurobipy` must be importable. Academic licences are available free from gurobi.com.

Install Python dependencies:
```bash
pip install -r requirements.txt
```

---

## Reproducing the paper results

All experiments in the paper can be reproduced with a single command:

```bash
python reproduce_paper.py
```

This runs five experiment scripts in sequence (~30 min on a modern laptop, single thread):

| Experiment | Script | Paper table | ~Time |
|---|---|---|---|
| LP baselines (§3.1/3.2) | `run_lp_baselines.py` | tab:popsyn_lp, tab:transport_lp | 2 min |
| Stage-contribution ablation (§3.3) | `run_ablation_setcover.py` | tab:ablation | 8 min |
| OR-Library benchmarks (§3.3) | `run_orlib_scp.py` | tab:orlib | 1 min |
| MIPFocus=1 comparison (§3.3) | `run_mipfocus_comparison.py` | (response letter) | 5 min |
| Extended warm-start (§3.3.3) | `run_warmstart_extended.py` | tab:hybrid_vs_mip | 15 min |

Each script can also be run individually:
```bash
python run_ablation_setcover.py
python run_orlib_scp.py
# etc.
```

Output CSVs and summary TXT files are written to `experiments/<name>/` with a
timestamp in the filename.

### OR-Library data

The OR-Library set-cover instances (scp41–scp49) must be present in `data/orlib_scp/`.
Download from:
```
https://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html
```
Place the files as `data/orlib_scp/scp41.txt` … `scp49.txt`.
The `reproduce_paper.py` script will check for these and skip the OR-Library
experiment with a warning if they are missing.

---

## Repository structure

```
reproduce_paper.py          # Master runner — reproduces all paper results
run_lp_baselines.py         # §3.1/3.2: LP baseline comparison
run_ablation_setcover.py    # §3.3.2: Stage 1-2 / 1-3 / 1-4 ablation
run_orlib_scp.py            # §3.3.2: OR-Library scp41-scp49 benchmarks
run_mipfocus_comparison.py  # §3.3.2: Gurobi MIPFocus=1 comparison
run_warmstart_extended.py   # §3.3.3: Extended warm-start (120s, dual bounds)

mip_hybrid/
  apps/
    synth_setcover.py       # Core set-cover solver (Stages 1-4)
    population_transport.py # Population synthesis and transportation solvers
  analysis/                 # Table generation utilities
  runners/                  # Legacy runners for original paper tables
  core/, io/                # Supporting modules

data/
  orlib_scp/                # Place OR-Library .txt files here
experiments/
  ablation/                 # Output from run_ablation_setcover.py
  lp_baselines/             # Output from run_lp_baselines.py
  orlib_scp/                # Output from run_orlib_scp.py
  mipfocus/                 # Output from run_mipfocus_comparison.py
  warmstart_extended/       # Output from run_warmstart_extended.py
```

---

## Skipping or selecting experiments

```bash
# Run only the ablation and OR-Library experiments
python reproduce_paper.py --only ablation orlib

# Run everything except the warm-start (longest experiment)
python reproduce_paper.py --skip warmstart
```

---

## Key parameters

All experiment scripts use the parameter settings reported in the paper:

| Parameter | Value | Role |
|---|---|---|
| Temperature schedule | τ ∈ {0.5, 0.2, 0.1} | Annealing stages |
| IPF sweeps | 50 per stage | Convergence |
| Restricted MIP budget | 1s | Stage 3 polish time |
| Pool size | top 30% | Stage 3 variable pool |
| MH steps | 150 | Stage 4 iterations |
| MH temperature | 0.1 | Stage 4 acceptance |
| Threads | 1 | Single-threaded throughout |

---

## Citation

If you use this code, please cite:

```
Rich, J. (2025). RICH: A Rapid Information-Theoretic Hybrid Algorithm for
Mixed-Integer Optimization. Mathematical Programming Computation (under review).
```
