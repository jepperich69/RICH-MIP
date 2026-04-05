# -*- coding: utf-8 -*-
"""
Optimized edition of SETCOVER_RAIL_application_generator_A_v2.py
- Adds fast incidence indices and vectorized kernels (no pandas anywhere).
- Keeps the public CLI and behavior intact.
"""
#!/usr/bin/env python3

import argparse, time, csv, sys, math, json, datetime as dt, os
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
try:
    from scipy.sparse import csr_matrix
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
import heapq
import os as _os
from types import SimpleNamespace   # ADD


# ---------- Solver selection + backend choice ----------
import os

def _env_float(*names):
    for n in names:
        v = os.getenv(n)
        if v not in (None, ""):
            try:
                return float(v)
            except ValueError:
                pass
    return None

def _env_int(*names):
    f = _env_float(*names)
    return int(f) if f is not None else None

def selected_solver() -> str:
    """
    User choice from CLI: 'cbc' or 'gurobi'.
    main() should set: os.environ['MIP_SOLVER'] = args.solver
    """
    return os.getenv("MIP_SOLVER", "cbc").lower()

# In mip_hybrid/apps/rail_setcover.py
# Replace the existing choose_backend function with this:

def choose_backend() -> str:
    """Choose backend based on solver preference"""
    solver_name = selected_solver()
    if solver_name == "gurobi":
        return "pulp"  # Route to PuLP for Gurobi access
    else:
        return "ortools"  # Use OR-Tools for CBC
    
# ---------- Lightweight profiler ----------
from contextlib import contextmanager
from time import perf_counter
from collections import defaultdict

class RunProfiler:
    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self.t = defaultdict(float)         # cumulative times per label
        self.events = []                    # fine-grained timeline
        self.counters = defaultdict(int)    # simple counters
        self.meta = {}
    @contextmanager
    def section(self, name, **meta):
        if not self.enabled:
            yield
            return
        t0 = perf_counter()
        try:
            yield
        finally:
            dt = perf_counter() - t0
            self.t[name] += dt
            ev = {"label": name, "dt": dt}
            if meta:
                ev.update(meta)
            self.events.append(ev)
    def add(self, name, dt, **meta):
        if not self.enabled:
            return
        self.t[name] += dt
        ev = {"label": name, "dt": dt}
        if meta:
            ev.update(meta)
        self.events.append(ev)
    def count(self, name, inc=1):
        if not self.enabled:
            return
        self.counters[name] += inc
    def to_dict(self):
        return {
            "times": dict(self.t),
            "counters": dict(self.counters),
            "events": self.events,
            "meta": self.meta,
        }

# ---------- Instance ----------
@dataclass
class SetCoverInstance:
    A: List[List[int]]   # element -> list of sets covering it
    c: np.ndarray        # costs (m,)
    n: int               # number of elements
    m: int               # number of sets
    k: int = 1           # coverage requirement (>=k)
    # Fast incidence views (built once via build_fast_views)
    _row_ptr: Optional[np.ndarray] = None
    _col_idx: Optional[np.ndarray] = None
    _row_ids: Optional[np.ndarray] = None
    _rows_of_col: Optional[List[np.ndarray]] = None

# ----- ONE-GO SPYDER CONFIG -----
RUN_MODE = "auto"
RAIL_DEFAULT = r"C:\Users\rich\OneDrive - Danmarks Tekniske Universitet\JR\Publikationer\MIP hybrid\rail582\rail582"


# io_dump.py
import json, numpy as np

# io_dump.py
import json, os, numpy as np

def _rows_of_col_from_A(A, m):
    """
    Build rows_of_col as a list of np.array of row indices covered by each column j.
    Works from A: rows[i] -> list of columns j that cover i. Returns length-m list.
    """
    roc = [list() for _ in range(m)]
    for i, cols in enumerate(A):
        # ensure python ints, tolerate np.int64 in cols
        for j in map(int, cols):
            roc[j].append(i)
    # convert to sorted unique arrays
    return [np.array(sorted(set(rr)), dtype=int) for rr in roc]

def dump_instance_json(inst, path, controller="polish", pool=None, x_init=None,
                       time_limit=3.0, solver="highs", threads=0,
                       index_base="auto"):
    """
    Write a JSON instance for the Julia runner.

    index_base: "auto" | 0 | 1
      - "auto": write indices exactly as stored in inst.A / pool (no remapping)
      - 0 or 1: force all indices to that base (will shift if needed)
    """
    # ensure folder exists
    _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)

    n = int(inst.n); m = int(inst.m); k = int(inst.k)

    def _detect_base_rows(rows):
        # Look for the first non-empty row
        for r in rows:
            if r:
                return 0 if min(map(int, r)) == 0 else 1
        return 1

    def _shift_list(lst, from_base, to_base):
        if from_base == to_base: 
            return list(map(int, lst))
        shift = (to_base - from_base)
        return [int(x) + shift for x in lst]

    payload = {
        "n": n, "m": m, "k": k,
        "controller": controller,
        "solver": str(solver),
        "time_limit": float(time_limit),
        "threads": int(threads),
    }

    if controller == "polish":
        assert pool is not None, "pool is required for polish"
        pool = list(map(int, pool))

        # Build rows_of_col map robustly
        have_cache = (getattr(inst, "_rows_of_col", None) is not None)
        if have_cache:
            roc_list = inst._rows_of_col
        else:
            roc_list = _rows_of_col_from_A(inst.A, m)

        # Determine base if auto: infer from pool (else from rows)
        if index_base == "auto":
            base_pool = 0 if pool and min(pool) == 0 else 1
            base = base_pool
        else:
            base = int(index_base)

        # Optional remap pool/rows to desired base
        if index_base != "auto":
            # detect current base from pool or from first roc entry
            cur_base = 0 if pool and min(pool) == 0 else 1
            pool = _shift_list(pool, cur_base, base)

        rows_of_col = {}
        for j in pool:
            rr = roc_list[j]
            # rr may be np.ndarray or list; cast to Python ints and sort/unique
            if not isinstance(rr, np.ndarray):
                rr = np.array(list(map(int, rr)), dtype=int)
            rr = np.unique(rr)
            if index_base != "auto":
                # detect rr base from content
                cur_base_rr = 0 if rr.size > 0 and rr.min() == 0 else 1
                if cur_base_rr != base:
                    rr = rr + (base - cur_base_rr)
            rows_of_col[int(j)] = rr.astype(int).tolist()

        payload["pool"] = list(map(int, pool))
        payload["cost_pool"] = [float(inst.c[j]) for j in pool]
        payload["rows_of_col"] = rows_of_col

        if x_init is not None:
            # x_init may be 0/1 array for the full m; extract entries for pool
            if len(x_init) == m:
                mip_start_pool = [int(x_init[j]) for j in pool]
            else:
                # assume already aligned with pool
                mip_start_pool = [int(v) for v in x_init]
                assert len(mip_start_pool) == len(pool), "x_init length must match pool"
            payload["mip_start_pool"] = mip_start_pool

    else:
        # Full data for mip_full / greedy_round
        rows = [list(map(int, cols)) for cols in inst.A]
        # sort & dedup rows to be safe
        rows = [sorted(set(r)) for r in rows]

        if index_base == "auto":
            base = _detect_base_rows(rows)
        else:
            base = int(index_base)

        if index_base != "auto":
            # detect current base
            cur_base = _detect_base_rows(rows)
            if cur_base != base:
                rows = [_shift_list(r, cur_base, base) for r in rows]

        payload["rows"] = rows
        payload["cost"] = [float(v) for v in np.asarray(inst.c).reshape(-1)]

    with open(path, "w") as f:
        json.dump(payload, f)


def round_cover_dual(inst, x_frac: np.ndarray, y: np.ndarray):
    _ensure_rows_of_col(inst)
    n, m, k = inst.n, inst.m, inst.k
    c = inst.c

    # 1) take all columns with r_j <= 0 immediately
    r = c - np.add.reduceat(y[inst._row_ids], inst._col_idx) if hasattr(inst, "_row_ids") else \
        c - _reduced_by_scatter(inst, y)  # fallback if you don't have _row_ids/_col_idx
    x = np.zeros(m, dtype=np.int8)
    take = (r <= 0.0)
    if take.any():
        x[take] = 1

    # 2) coverage counts from current x
    cover_cnt = np.zeros(n, dtype=np.int32)
    on = np.where(x == 1)[0]
    for j in on:
        rows = inst._rows_of_col[j]
        if rows.size: cover_cnt[rows] += 1

    # 3) build lazy heap of ( -gain/cost, stamp, j ). gain = newly satisfied unit deficits if we add j
    # deficits are rows with cover_cnt < k
    deficit = (cover_cnt < k)
    if not deficit.any() and on.size > 0:
        cost = float(inst.c @ x)
        return x.astype(np.int8), cost, float(cover_cnt.min()), True

    stamp = np.zeros(m, dtype=np.int32)
    cur_stamp = 1
    heap = []

    def gain_of_col(j):
        rows = inst._rows_of_col[j]
        if rows.size == 0: return 0
        # rows that are still below k become more covered by 1 if we pick j
        return int(np.count_nonzero(deficit[rows]))

    # initialize gains only for promising columns
    # (filter out columns already on or with zero potential gain)
    for j in range(m):
        if x[j]: continue
        g = gain_of_col(j)
        if g > 0:
            heapq.heappush(heap, (-g / max(c[j], 1e-12), cur_stamp, j))
            stamp[j] = cur_stamp
    cur_stamp += 1

    # 4) greedy add until all covered >= k
    while deficit.any():
        if not heap:
            # fall back: add the cheapest column that hits any deficit row
            # (should be rare with the heap)
            best_j, best_score = -1, -1.0
            for j in range(m):
                if x[j]: continue
                g = gain_of_col(j)
                if g <= 0: continue
                sc = g / max(c[j], 1e-12)
                if sc > best_score:
                    best_score, best_j = sc, j
            if best_j == -1:
                break
            j = best_j
        else:
            sc_neg, st, j = heapq.heappop(heap)
            # lazy check: if stale, recompute and push back
            if stamp[j] != st:
                g = gain_of_col(j)
                if g > 0:
                    heapq.heappush(heap, (-g / max(c[j], 1e-12), stamp[j], j))
                continue

        # take j
        x[j] = 1
        rows = inst._rows_of_col[j]
        if rows.size:
            pre = (cover_cnt[rows] < k)
            cover_cnt[rows] += 1
            # only rows that changed from deficit→non-deficit affect gains
            changed = rows[pre & (cover_cnt[rows] >= k)]
            if changed.size:
                # for all columns touching these rows, recompute gain lazily
                touched_cols = set()
                for i in changed:
                    for jj in inst.A[i]:
                        if x[jj]: continue
                        touched_cols.add(jj)
                cur_stamp += 1
                for jj in touched_cols:
                    stamp[jj] = cur_stamp
                    g = gain_of_col(jj)
                    if g > 0:
                        heapq.heappush(heap, (-g / max(c[jj], 1e-12), cur_stamp, jj))

        deficit = (cover_cnt < k)

    feas = bool(cover_cnt.min() >= k)
    return x.astype(np.int8), float(inst.c @ x), float(cover_cnt.min()), feas

# helper when _row_ids/_col_idx aren't available (simple scatter-add)
def _reduced_by_scatter(inst, y):
    acc = np.zeros(inst.m, dtype=float)
    for i, cols in enumerate(inst.A):
        if not cols: continue
        acc[cols] += y[i]
    return acc

def resolve_rail_path(cli_val: str | None) -> str | None:
    if cli_val and _os.path.isfile(cli_val):
        return cli_val
    envp = _os.environ.get("RAIL_PATH", "").strip()
    if envp and _os.path.isfile(envp):
        return envp
    if RAIL_DEFAULT and _os.path.isfile(RAIL_DEFAULT):
        return RAIL_DEFAULT
    return None

# ---------- Fast incidence builder ----------
def build_fast_views(inst: SetCoverInstance):
    """Build CSR-like arrays + rows_of_col once and attach to instance."""
    n, m = inst.n, inst.m
    lengths = np.fromiter((len(cols) for cols in inst.A), dtype=np.int64, count=n)
    row_ptr = np.empty(n+1, dtype=np.int64)
    row_ptr[0] = 0
    np.cumsum(lengths, out=row_ptr[1:])
    nnz = int(row_ptr[-1])

    col_idx = np.empty(nnz, dtype=np.int32)
    p = 0
    for cols in inst.A:
        L = len(cols)
        if L:
            col_idx[p:p+L] = np.asarray(cols, dtype=np.int32)
            p += L

    # row index per nonzero for fast scatter/reductions
    row_ids = np.repeat(np.arange(n, dtype=np.int32), lengths)

    # rows touched by each column (reverse incidence)
    rows_of_col = [[] for _ in range(m)]
    for i, cols in enumerate(inst.A):
        for j in cols:
            rows_of_col[j].append(i)
    rows_of_col = [np.asarray(v, dtype=np.int32) if len(v) else np.empty(0, dtype=np.int32)
                   for v in rows_of_col]

    inst._row_ptr = row_ptr
    inst._col_idx = col_idx
    inst._row_ids = row_ids
    inst._rows_of_col = rows_of_col
    return inst

# --- loader of RAIL
def load_orlib_rail(path: str, k: int = 1) -> SetCoverInstance:
    import io
    with open(path, "r") as f:
        toks = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks += line.split()
            if len(toks) >= 2:
                break
        if len(toks) < 2:
            raise ValueError("Header line with 'm n' not found.")
        m_rows, n_cols = int(toks[0]), int(toks[1])

    A = [[] for _ in range(m_rows)]
    c = np.zeros(n_cols, dtype=float)

    with open(path, "r") as f:
        header_read = False
        buf = []
        def consume_ints():
            nonlocal buf
            while True:
                if buf:
                    v = buf.pop(0)
                    return int(v)
                line = f.readline()
                if not line:
                    return None
                line = line.strip()
                if not line:
                    continue
                buf += line.split()
        _m = consume_ints(); _n = consume_ints()
        if _m != m_rows or _n != n_cols:
            raise ValueError("Header mismatch while rereading file.")
        for j in range(n_cols):
            cj = consume_ints()
            rj = consume_ints()
            if cj is None or rj is None:
                raise ValueError(f"Unexpected EOF while reading column {j+1}.")
            c[j] = float(cj)
            for _ in range(rj):
                i = consume_ints()
                if i is None:
                    raise ValueError(f"Unexpected EOF inside column {j+1}.")
                i0 = i - 1
                if not (0 <= i0 < m_rows):
                    raise ValueError(f"Row index out of range: {i} in column {j+1}.")
                A[i0].append(j)

    inst = SetCoverInstance(A=A, c=c, n=m_rows, m=n_cols, k=k)
    
    # Add this analysis:
    coverage_per_row = [len(inst.A[i]) for i in range(inst.n)]
    sets_per_element = np.array(coverage_per_row)
    print(f"RAIL582: coverage per row mean={sets_per_element.mean():.1f}, "
          f"density={sets_per_element.sum()/(inst.n * inst.m):.4f}")
    
    return build_fast_views(inst)

def make_set_cover(n: int, m: int, alpha: float = 1.8, seed: Optional[int] = None, k: int = 1) -> SetCoverInstance:
    rng = np.random.default_rng(seed if seed is not None else 2025)
    sizes = np.maximum(1, (rng.pareto(alpha, size=m) * (n / 20)).astype(int))
    S = []
    for j in range(m):
        S_j = rng.choice(n, size=min(n, sizes[j]), replace=False).tolist()
        S.append(S_j)
    c = 0.5 + rng.random(m)
    A = [[] for _ in range(n)]
    for j, S_j in enumerate(S):
        for i in S_j:
            A[i].append(j)
    for i in range(n):
        while len(A[i]) < k:
            jj = rng.integers(0, m)
            if jj not in A[i]:
                A[i].append(jj)
    inst = SetCoverInstance(A=A, c=c, n=n, m=m, k=k)
    return build_fast_views(inst)

# ---------- LP lower bound ----------
# ---------- LP (relaxation) solve ----------
def solve_lp(inst: SetCoverInstance, timelimit: float = 60.0):
    """
    Solve the LP relaxation of set cover:
      min  sum_j c_j x_j
      s.t. sum_{j in A[i]} x_j >= k  for all i
           0 <= x_j <= 1
    Backend selection mirrors solve_mip():
      - If --solver cbc: try OR-Tools (GLOP) first, else PuLP+CBC
      - If --solver gurobi: use PuLP+GUROBI_CMD (gurobi_cl)
    """
    backend = choose_backend()
    t0 = time.time()

    if backend == 'ortools':
        # Prefer OR-Tools' LP solver (GLOP). Fallback to CBC if GLOP not available.
        try:
            from ortools.linear_solver import pywraplp
        except Exception:
            backend = 'pulp'  # fall through to PuLP path
        else:
            solver = pywraplp.Solver.CreateSolver('GLOP')
            if solver is None:
                # as a fallback, use CBC through the same API
                solver = pywraplp.Solver.CreateSolver('CBC')
            # time limit (ms) if supported
            try:
                if timelimit is not None:
                    solver.SetTimeLimit(int(max(0, timelimit) * 1000))
            except Exception:
                pass

            x = [solver.NumVar(0.0, 1.0, f"x_{j}") for j in range(inst.m)]

            for i in range(inst.n):
                ct = solver.RowConstraint(float(inst.k), solver.infinity(), f"cover_{i}")
                for j in inst.A[i]:
                    ct.SetCoefficient(x[j], 1.0)

            obj = solver.Objective()
            for j in range(inst.m):
                obj.SetCoefficient(x[j], float(inst.c[j]))
            obj.SetMinimization()

            result = solver.Solve()
            t = time.time() - t0
            objv = obj.Value()
            return dict(obj=float(objv), time=float(t), status=result, backend='ortools')

    # ---- PuLP path (CBC or GUROBI depending on selected_solver) ----
    import pulp

    prob = pulp.LpProblem("SetCoverLP", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{j}", lowBound=0.0, upBound=1.0, cat='Continuous') for j in range(inst.m)]

    for i in range(inst.n):
        prob += pulp.lpSum([x[j] for j in inst.A[i]]) >= float(inst.k)

    prob += pulp.lpSum(float(inst.c[j]) * x[j] for j in range(inst.m))

    sel = selected_solver()  # 'cbc' or 'gurobi'

# --- GUROBI via PuLP: pass params via options (no 'epgap' kw) ---
# --- GUROBI via PuLP (no 'epgap' kwarg; use options list) ---
    if sel == "gurobi":
        import os
        import pulp
    
        # Optional overrides from environment (easy to tweak from Spyder)
        gap_env = os.getenv("GRB_MIPGAP") or os.getenv("MIP_GAP")     # e.g., "0.10" for 10%
        thr_env = os.getenv("GRB_THREADS") or os.getenv("THREADS")    # e.g., "1"
    
        opt_list = []
        if gap_env is not None and str(gap_env).strip() != "":
            try:
                opt_list.append(("MIPGap", float(gap_env)))
            except Exception:
                print(f"[warn] ignoring invalid GRB_MIPGAP/MIP_GAP value: {gap_env!r}")
        if thr_env is not None and str(thr_env).strip() != "":
            try:
                opt_list.append(("Threads", int(thr_env)))
            except Exception:
                print(f"[warn] ignoring invalid GRB_THREADS/THREADS value: {thr_env!r}")
    
        solver = pulp.GUROBI_CMD(
            msg=False,
            timeLimit=int(timelimit) if timelimit is not None else None,
            options=opt_list
        )
        backend_tag = 'pulp:gurobi'
    else:
        thr_env = os.getenv("GRB_THREADS") or os.getenv("THREADS")
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=int(timelimit, threads=(int(thr_env) if thr_env not in (None, "") else None)) if timelimit is not None else None)
        backend_tag = 'pulp:cbc'
    
    prob.solve(solver)
    t = time.time() - t0
    objv = float(pulp.value(prob.objective)) if pulp.value(prob.objective) is not None else float("nan")
    return dict(obj=objv, time=float(t), status=pulp.LpStatus[prob.status], backend=backend_tag)


def solve_mip(inst: SetCoverInstance, timelimit: float = 60.0, 
              gurobi_time_limit: float = None, gurobi_gap_limit: float = None,
              track_anytime: bool = False):
    print(f"[SOLVE_MIP] track_anytime={track_anytime}, solver={selected_solver()}, backend={choose_backend()}")  # ADD THIS

    """
    Solve the set cover MIP with enhanced controls for fair comparison.
    
    Parameters:
    - timelimit: Base time limit (used if gurobi_time_limit not specified)
    - gurobi_time_limit: Specific time limit for Gurobi (overrides timelimit)
    - gurobi_gap_limit: Gap limit for early termination (e.g., 0.01 for 1%)
    - track_anytime: If True, track incumbent progression over time
    """
    backend = choose_backend()
    anytime_log = [] if track_anytime else None
    
    # Use specific Gurobi limits if provided, otherwise fall back to timelimit
    actual_time_limit = gurobi_time_limit if gurobi_time_limit is not None else timelimit

    # ---- OR-Tools CBC path ----
    if backend == 'ortools':
        try:
            from ortools.linear_solver import pywraplp
        except Exception:
            backend = 'pulp'
        else:
            solver = pywraplp.Solver.CreateSolver('CBC')
            if solver is None:
                raise RuntimeError("Failed to create OR-Tools CBC solver")

            if actual_time_limit is not None:
                solver.SetTimeLimit(int(max(0, actual_time_limit) * 1000))

            x = [solver.IntVar(0, 1, f"x_{j}") for j in range(inst.m)]

            for i in range(inst.n):
                ct = solver.RowConstraint(float(inst.k), solver.infinity(), f"cover_{i}")
                for j in inst.A[i]:
                    ct.SetCoefficient(x[j], 1.0)

            obj = solver.Objective()
            for j in range(inst.m):
                obj.SetCoefficient(x[j], float(inst.c[j]))
            obj.SetMinimization()

            t0 = time.time()
            result = solver.Solve()
            t = time.time() - t0

            xv = np.array([int(var.solution_value() > 0.5) for var in x], dtype=int)
            objv = obj.Value()
            try:
                bound = solver.Objective().BestBound()
                gap = abs(objv - bound) / max(1e-9, abs(objv))
            except Exception:
                gap = None

            return dict(x=xv, obj=float(objv), time=float(t),
                        gap=gap, status=result, backend='ortools:cbc',
                        anytime_log=anytime_log)

    # ---- PuLP path (handles both CBC and Gurobi) ----
    import pulp
    
    solver_name = selected_solver()
    
    # Route to direct Gurobi if anytime tracking requested
    if track_anytime and solver_name == "gurobi":
        import os
        mipgap = None
        threads = None
        
        gap_env = os.getenv("GRB_MIPGAP") or os.getenv("MIP_GAP")
        thr_env = os.getenv("GRB_THREADS") or os.getenv("THREADS")
        
        if gurobi_gap_limit is not None:
            mipgap = float(gurobi_gap_limit)
        elif gap_env not in (None, ""):
            try:
                mipgap = float(gap_env)
            except Exception:
                pass
        
        if thr_env not in (None, ""):
            try:
                threads = int(thr_env)
            except Exception:
                pass
        
        return _solve_mip_gurobi_direct(inst, actual_time_limit, mipgap, threads, gurobi_gap_limit)
    
    # Standard PuLP path (no anytime tracking)
    prob = pulp.LpProblem("SetCover", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{j}", lowBound=0, upBound=1, cat='Binary') for j in range(inst.m)]

    # Coverage constraints
    for i in range(inst.n):
        prob += pulp.lpSum([x[j] for j in inst.A[i]]) >= float(inst.k)

    # Objective
    prob += pulp.lpSum(float(inst.c[j]) * x[j] for j in range(inst.m))

    # Configure solver
    import os
    
    mipgap = None
    threads = None
    
    if solver_name == "gurobi":
        gap_env = os.getenv("GRB_MIPGAP") or os.getenv("MIP_GAP")
        thr_env = os.getenv("GRB_THREADS") or os.getenv("THREADS")
    
        options = []
        
        if gurobi_gap_limit is not None:
            mipgap = float(gurobi_gap_limit)
            options.append(("MIPGap", mipgap))
        elif gap_env not in (None, ""):
            try:
                mipgap = float(gap_env)
                options.append(("MIPGap", mipgap))
            except Exception:
                pass
        
        if thr_env not in (None, ""):
            try:
                threads = int(thr_env)
                options.append(("Threads", threads))
            except Exception:
                pass

        solver = pulp.GUROBI_CMD(
            msg=False,
            timeLimit=int(actual_time_limit) if actual_time_limit is not None else None,
            options=options,
        )
        backend_tag = "pulp:gurobi"
    else:
        thr_env = os.getenv("GRB_THREADS") or os.getenv("THREADS")
        solver = pulp.PULP_CBC_CMD(
            msg=False,
            timeLimit=int(actual_time_limit) if actual_time_limit is not None else None,
            threads=(int(thr_env) if thr_env not in (None, "") else None)
        )
        backend_tag = "pulp:cbc"
    
    t0 = time.time()
    prob.solve(solver)
    t = time.time() - t0
    xv = np.array([int(v.value() > 0.5) for v in x], dtype=int)
    objv = float(pulp.value(prob.objective))
    
    return dict(
        x=xv,
        obj=objv,
        time=float(t),
        gap=None,
        status=pulp.LpStatus[prob.status],
        backend=backend_tag,
        anytime_log=anytime_log,
    )


def _solve_mip_gurobi_direct(inst: SetCoverInstance, time_limit: float, 
                            mip_gap: float, threads: int, gap_limit: float):
    """
    Direct Gurobi solve with anytime incumbent tracking.
    Only called when track_anytime=True and using Gurobi.
    """
    try:
        import gurobipy as gp
        import time
    except ImportError:
        return dict(x=np.zeros(inst.m), obj=float('inf'), time=0.0, gap=None, 
                   status='Error', backend='gurobi:unavailable', anytime_log=[])

    anytime_log = []
    
    # Callback to track incumbent progression
    def callback(model, where):
        if where == gp.GRB.Callback.MIPSOL:
            cur_obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
            cur_time = time.time() - start_time
            anytime_log.append((cur_time, cur_obj, 'gurobi_incumbent'))
    
    t0 = time.time()
    
    # Build model
    env = gp.Env(empty=True)
    
    # Set parameters
    if time_limit is not None:
        env.setParam("TimeLimit", float(time_limit))
    if gap_limit is not None:
        env.setParam("MIPGap", float(gap_limit))
    elif mip_gap is not None:
        env.setParam("MIPGap", float(mip_gap))
    if threads is not None:
        env.setParam("Threads", int(threads))
    
    env.setParam("OutputFlag", 0)
    env.start()
    
    model = gp.Model(env=env)
    
    # Variables
    x_vars = model.addMVar(shape=inst.m, vtype=gp.GRB.BINARY, name="x")
    
    # Constraints
    for i in range(inst.n):
        idx = [j for j in inst.A[i]]
        if idx:
            model.addConstr(x_vars[idx].sum() >= float(inst.k))
    
    # Objective
    model.setObjective(inst.c @ x_vars, gp.GRB.MINIMIZE)
    
    # Optimize with callback
    start_time = time.time()
    model.optimize(callback)
    total_time = time.time() - t0
    
    if model.Status not in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT):
        return dict(x=np.zeros(inst.m), obj=float('inf'), time=total_time, 
                   gap=None, status=f'Status_{model.Status}', 
                   backend='gurobi:direct', anytime_log=anytime_log)
    
    # Extract solution
    xv = np.array([int(v.X > 0.5) for v in x_vars], dtype=int)
    objv = float(model.ObjVal)
    
    try:
        bound = float(model.ObjBound)
        gap = abs(objv - bound) / max(1e-9, abs(objv))
    except Exception:
        gap = None
    
    return dict(
        x=xv,
        obj=objv,
        time=total_time,
        gap=gap,
        status='Optimal' if model.Status == gp.GRB.OPTIMAL else 'TimeLimit',
        backend='gurobi:direct',
        anytime_log=anytime_log,
    )# ---------- IPF (row-wise) with dual tracking & annealing ----------

def _cov_from_x(inst: SetCoverInstance, x: np.ndarray) -> np.ndarray:
    """Row cover sums using vectorized scatter: cov[i] = sum_{j in A[i]} x[j]."""
    rp, ci, ri = inst._row_ptr, inst._col_idx, inst._row_ids
    # contribution per nonzero = x[col]
    contrib = x[ci]
    cov = np.zeros(inst.n, dtype=float)
    np.add.at(cov, ri, contrib)
    return cov

def ipf_rowwise_with_dual(inst: SetCoverInstance, tau: float, iters: int, tol: float, clip_one: bool, x0: Optional[np.ndarray] = None):
    """
    Faster row-wise KL projection using prebuilt CSR-like arrays.
    Returns x, y, smooth_obj, wall_time, lin_cost, cov_min, cov_avg.
    """
    t0 = time.time()
    q = np.exp(-inst.c / max(tau, 1e-12))
    x = q.copy() if x0 is None else x0.copy()
    y = np.zeros(inst.n, dtype=float)

    rhs = float(inst.k)
    rp, ci = inst._row_ptr, inst._col_idx

    for sweep in range(iters):
        max_violation = 0.0
        # loop rows; inner ops are vectorized on slices
        for i in range(inst.n):
            start, end = int(rp[i]), int(rp[i+1])
            if start == end:
                continue
            cols = ci[start:end]
            s = float(x[cols].sum())
            if s < rhs - tol:
                alpha = rhs / max(s, 1e-12)
                x[cols] *= alpha
                y[i] += tau * math.log(alpha)
                max_violation = max(max_violation, rhs - s)
        if clip_one:
            np.minimum(x, 1.0, out=x)
        if max_violation <= tol:
            break

    cov = _cov_from_x(inst, x)
    lin_cost = float(inst.c @ x)
    smooth_obj = lin_cost + tau * float(np.sum(x * (np.log(np.maximum(x,1e-12)) - 1.0)))
    return x, y, smooth_obj, time.time() - t0, lin_cost, float(cov.min()), float(cov.mean())

def entropy_relax_setcover_ipf_anneal(inst: SetCoverInstance, tau: float = 0.2,
                                      iters: int = 50, tol: float = 1e-3,
                                      clip_one: bool = True, tau_schedule: Optional[List[float]] = None,
                                      extra_sweeps: int = 120):
    """
    Run row-wise IPF with an optional tau schedule (coarse-to-fine).
    Warm-start between temperatures and do a final polish.
    """
    taus = tau_schedule if (tau_schedule and len(tau_schedule) > 0) else [tau]
    x = None; y = None; total_time = 0.0; lin_cost = None; smooth_obj = None; cov_min = None; cov_avg = None
    prev_t = None
    for t in taus:
        if x is None:
            x0 = None
        else:
            reweight = np.exp(-(1.0/max(t,1e-12) - 1.0/max(prev_t,1e-12)) * inst.c)
            x0 = np.clip(x * reweight, 1e-18, 1.0 if clip_one else np.inf)
        x, y, smooth_obj, dt, lin_cost, cov_min, cov_avg = ipf_rowwise_with_dual(
            inst, tau=t, iters=iters, tol=tol, clip_one=clip_one, x0=x0
        )
        total_time += dt
        prev_t = t

    sweeps = 0
    while cov_min < inst.k - tol and sweeps < extra_sweeps:
        x, y, smooth_obj, dt, lin_cost, cov_min, cov_avg = ipf_rowwise_with_dual(
            inst, tau=taus[-1], iters=10, tol=tol*0.5, clip_one=clip_one, x0=x
        )
        total_time += dt
        sweeps += 10

    return x, y, smooth_obj, total_time, lin_cost, cov_min, cov_avg

# ---------- Rounding (dual-guided + prune) ----------
def compute_reduced_costs(inst: SetCoverInstance, y: np.ndarray, prof: RunProfiler | None = None):
    t0 = perf_counter() if (prof and getattr(prof, "enabled", False)) else None
    acc = np.zeros(inst.m, dtype=float)
    np.add.at(acc, inst._col_idx, y[inst._row_ids])
    out = inst.c - acc
    if t0 is not None:
        prof.add('reduced_costs', perf_counter() - t0)
    return out

def round_cover_dual(inst: SetCoverInstance, x_frac: np.ndarray, y: np.ndarray):
    n, m, k = inst.n, inst.m, int(inst.k)
    S = inst._rows_of_col
    r = compute_reduced_costs(inst, y)

    x_int = np.zeros(m, dtype=int)
    cover_cnt = np.zeros(n, dtype=int)
    # take all non-positive reduced cost columns
    nz = np.where(r <= 1e-12)[0]
    if nz.size:
        x_int[nz] = 1
        for j in nz:
            rows = S[j]
            if rows.size:
                cover_cnt[rows] += 1

    def gain_of(j):
        if x_int[j] == 1: return 0
        rows = S[j]
        if rows.size == 0: return 0
        deficit = (cover_cnt[rows] < k).sum()
        return int(deficit)

    safety = 0
    while np.any(cover_cnt < k) and safety < 5*m:
        best, best_ratio = -1, float('inf')
        # consider only columns that touch some under-covered row
        deficit_rows = np.where(cover_cnt < k)[0]
        cand = set()
        for i in deficit_rows:
            for j in inst.A[i]:
                if x_int[j] == 0:
                    cand.add(j)
        for j in cand:
            g = gain_of(j)
            if g > 0:
                eff_cost = max(1e-12, r[j])
                ratio = eff_cost / g
                if ratio < best_ratio:
                    best_ratio = ratio; best = j
        if best == -1:
            i0 = int(np.argmin(cover_cnt))
            cand2 = [j for j in inst.A[i0] if x_int[j]==0]
            if not cand2: break
            best = min(cand2, key=lambda jj: r[jj])
        x_int[best] = 1
        rows = S[best]
        if rows.size:
            need = cover_cnt[rows] < k
            cover_cnt[rows[need]] += 1
        safety += 1

    # prune
    improved = True
    on_cols = np.where(x_int==1)[0]
    while improved:
        improved = False
        for j in on_cols:
            if x_int[j] == 0: 
                continue
            rows = S[j]
            if rows.size == 0: 
                x_int[j] = 0
                continue
            # check if removing j violates any row
            if np.any(cover_cnt[rows] - 1 < k):
                continue
            x_int[j] = 0
            cover_cnt[rows] -= 1
            improved = True

    cov_min = float(cover_cnt.min()) if cover_cnt.size else 0.0
    feasible = bool(cov_min >= k)
    cost = float(inst.c @ x_int)
    return x_int, cost, cov_min, feasible

def round_cover_greedy(inst, x_frac: np.ndarray):
    _ensure_rows_of_col(inst)
    n, m, k = inst.n, inst.m, inst.k
    c = inst.c

    # initial ranking by cost/coverage weight from fractional solution
    # (use -x_frac/c as a proxy; avoids loops)
    score = x_frac / np.maximum(c, 1e-12)
    order = np.argsort(-score)  # best first

    x = np.zeros(m, dtype=np.int8)
    cover_cnt = np.zeros(n, dtype=np.int32)

    # 1) add columns until all rows reach k
    for j in order:
        if np.all(cover_cnt >= k):
            break
        rows = inst._rows_of_col[j]
        if rows.size == 0:
            continue
        # marginal gain test: does j hit any deficit row?
        if np.any(cover_cnt[rows] < k):
            x[j] = 1
            cover_cnt[rows] += 1

    feas = bool(cover_cnt.min() >= k)
    if not feas:
        # 2) fast repair: add cheapest columns that hit remaining deficits
        deficit_rows = np.where(cover_cnt < k)[0]
        needed = k - cover_cnt[deficit_rows]
        # simple loop over deficit rows, but each step uses vector ops
        for i in deficit_rows:
            need = k - cover_cnt[i]
            if need <= 0: continue
            # pick 'need' cheapest unused columns covering row i
            cands = [j for j in inst.A[i] if x[j] == 0]
            if cands:
                jj = np.argsort(c[cands])[:int(need)]
                for idx in jj:
                    j = cands[int(idx)]
                    x[j] = 1
                    rows = inst._rows_of_col[j]
                    if rows.size: cover_cnt[rows] += 1
        feas = bool(cover_cnt.min() >= k)

    return x.astype(np.int8), float(inst.c @ x), float(cover_cnt.min()), feas

def randomized_rounding_trials(inst, x_frac, trials: int = 20, thr_max: float = 0.6, rng=None):
    """Simple working version - just reduce trials instead of complex vectorization."""
    _ensure_rows_of_col(inst)
    n, m, k = inst.n, inst.m, inst.k
    c = inst.c
    if rng is None:
        rng = np.random.default_rng()

    best_x = None
    best_cost = float('inf')
    best_cov = 0
    best_feas = False

    # Simple approach: just do fewer trials efficiently
    p = np.clip(x_frac / max(x_frac.max(), 1e-9), 0.0, 1.0) * thr_max
    
    for t in range(trials):
        # Generate one trial at a time
        U = rng.random(m)
        x = (U < p).astype(np.int8)
        
        # Quick coverage check
        cover_cnt = np.zeros(n, dtype=np.int32)
        on = np.where(x == 1)[0]
        for j in on:
            rows = inst._rows_of_col[j]
            if rows.size: 
                cover_cnt[rows] += 1

        # Quick repair if needed
        safety = 0
        while np.any(cover_cnt < k) and safety < 100:
            deficit_rows = np.where(cover_cnt < k)[0]
            if deficit_rows.size == 0:
                break
            
            # Pick a random deficit row and cheapest column covering it
            i = deficit_rows[rng.integers(len(deficit_rows))]
            candidates = [j for j in inst.A[i] if x[j] == 0]
            if not candidates:
                break
                
            j = min(candidates, key=lambda jj: c[jj])
            x[j] = 1
            rows = inst._rows_of_col[j]
            if rows.size:
                cover_cnt[rows] += 1
            safety += 1

        feas = bool(cover_cnt.min() >= k)
        cost = float(c @ x)
        
        if (feas and cost < best_cost) or (not best_feas and feas) or (best_x is None):
            best_x, best_cost, best_cov, best_feas = x.copy(), cost, float(cover_cnt.min()), feas

    if best_x is None:
        return np.zeros(m, dtype=np.int8), float('inf'), 0.0, False
    return best_x.astype(np.int8), best_cost, best_cov, best_feas
def local_improve_drop_and_fix(inst, x_int, passes: int = 2, rng=None):
    _ensure_rows_of_col(inst)
    n, m, k = inst.n, inst.m, inst.k
    c = inst.c
    if rng is None:
        rng = np.random.default_rng()

    x = x_int.astype(np.int8).copy()
    # coverage counts once
    cover_cnt = np.zeros(n, dtype=np.int32)
    on = np.where(x == 1)[0]
    for j in on:
        rows = inst._rows_of_col[j]
        if rows.size: cover_cnt[rows] += 1

    for _ in range(max(1, passes)):
        order = on.copy()
        rng.shuffle(order)  # random order
        kept = []
        for j in order:
            rows = inst._rows_of_col[j]
            if rows.size == 0:
                x[j] = 0
                continue
            # if any row would drop below k -> keep j
            if np.any(cover_cnt[rows] <= k):
                kept.append(j)
                continue
            # safe to drop
            x[j] = 0
            cover_cnt[rows] -= 1
        on = np.array(kept, dtype=int)

    feas = bool(cover_cnt.min() >= k)
    return x.astype(np.int8), float(inst.c @ x), float(cover_cnt.min()), feas

def polish_restricted_mip_pulp(inst, x_init, r, timelimit=3.0, pool_frac=0.3):
    """
    Restricted MIP polishing using PuLP (CBC or Gurobi via GUROBI_CMD).
    Keeps the incumbent support plus a pool of best reduced-cost columns.
    """
    import time, os, numpy as np, pulp

    m = inst.m
    # keep current support + top-k (best reduced costs)
    keep = np.flatnonzero(np.asarray(x_init) > 0.5)
    r = np.asarray(r)
    k = max(1, int(pool_frac * m))
    pool = np.argsort(r)[:k]
    cols = np.unique(np.concatenate([keep, pool]))
    colset = set(int(j) for j in cols)

    prob = pulp.LpProblem("PolishRestricted", pulp.LpMinimize)
    xvar = [None]*m
    for j in range(m):
        if j in colset:
            xvar[j] = pulp.LpVariable(f"x_{j}", 0, 1, cat="Binary")

    # cover constraints, but only over available columns
    for i in range(inst.n):
        js = [j for j in inst.A[i] if (xvar[j] is not None)]
        if js:
            prob += pulp.lpSum(xvar[j] for j in js) >= float(inst.k)

    prob += pulp.lpSum(float(inst.c[j]) * xvar[j] for j in range(m) if xvar[j] is not None)

    # solver selection (same knob you’re already using)
    sel = (os.getenv("MIP_SOLVER", "cbc") or "cbc").lower()
    if sel == "gurobi":
        # pick optional overrides from env
        gap_env = os.getenv("GRB_MIPGAP") or os.getenv("MIP_GAP")
        thr_env = os.getenv("GRB_THREADS") or os.getenv("THREADS")
        opts = []
        if gap_env:  opts.append(("MIPGap", float(gap_env)))
        if thr_env:  opts.append(("Threads", int(thr_env)))
        solver = pulp.GUROBI_CMD(msg=False, timeLimit=int(timelimit), options=opts)
    else:
        thr_env = os.getenv("GRB_THREADS") or os.getenv("THREADS")
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=int(timelimit, threads=(int(thr_env) if thr_env not in (None, "") else None)))

    t0 = time.time()
    prob.solve(solver)
    T = time.time() - t0

    # build solution (fallback to incumbent if var not present)
    x_pol = np.array(
        [(xvar[j].value() > 0.5) if xvar[j] is not None else (x_init[j] > 0.5)
         for j in range(m)],
        dtype=int,
    )
    try:
        cost_pol = float(pulp.value(prob.objective))
    except Exception:
        cost_pol = float("inf")

    feas = (pulp.LpStatus[prob.status] in ("Optimal", "TimeLimit", "Not Solved", "Integer Feasible"))
    return x_pol, cost_pol, bool(feas), float(T)


# ---------- Restricted MIP polish ----------
def polish_restricted_mip(inst: SetCoverInstance, x_init: np.ndarray, r: np.ndarray,
                          timelimit=0.5, pool_frac=0.25,
                          prof: 'RunProfiler|None' = None):
    from ortools.linear_solver import pywraplp
    import time
    m = inst.m

    pool = set(np.where(x_init == 1)[0])
    k_pool = min(m, max(len(pool), int(pool_frac * m)))
    cand_by_r = np.argsort(r)[:k_pool]
    pool.update(cand_by_r.tolist())
    pool = sorted(pool)

    solver = pywraplp.Solver.CreateSolver('CBC')
    if solver is None:
        return x_init.copy(), float(inst.c @ x_init), False, 0.0

    try:
        solver.EnableOutput(False)
    except Exception:
        pass
    solver.SetTimeLimit(int(1000 * timelimit))

    x = {j: solver.IntVar(0, 1, f"x_{j}") for j in pool}

    for i in range(inst.n):
        ct = solver.RowConstraint(float(inst.k), solver.infinity(), "")
        for j in inst.A[i]:
            v = x.get(j)
            if v is not None:
                ct.SetCoefficient(v, 1.0)

    obj = solver.Objective()
    for j in pool:
        obj.SetCoefficient(x[j], float(inst.c[j]))
    obj.SetMinimization()

    try:
        solver.SetHint([x[j] for j in pool], [float(x_init[j]) for j in pool])
    except Exception:
        pass

    t0 = time.time()
    status = solver.Solve()
    t = time.time() - t0
    if prof and getattr(prof, "enabled", False):
        prof.add('polish_mip', t)

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return x_init.copy(), float(inst.c @ x_init), False, t

    x_new = np.zeros(m, dtype=int)
    for j in pool:
        if x[j].solution_value() > 0.5:
            x_new[j] = 1

    # quick feasibility check
    cover = np.zeros(inst.n, dtype=int)
    for i in range(inst.n):
        cover[i] = int(x_new[inst.A[i]].sum())
    feasible = bool(cover.min() >= inst.k)
    cost = float(inst.c @ x_new)
    return x_new, cost, feasible, t

# ---------- Benchmark harness ----------
def run_family(args):
    """
    Run one RAIL instance and return a single row of results (dict).
    The caller (main) is responsible for looping over --trials and writing CSV.
    """
    import numpy as np
    from time import perf_counter as _pc

    # --------- local stage timers (seconds) ---------
    _T = {
        "lp_solve": 0.0,
        "mip_solve": 0.0,
        "relax_total": 0.0,
        "round_rr_trials": 0.0,
        "round_main": 0.0,
        "improve_local": 0.0,
        "polish_mip": 0.0,
        "reduced_costs": 0.0,
    }
    _tall0 = _pc()  # global wall-time start

    if getattr(args, "seed", None) is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)

    anytime_log = []

    def _fmt(v):
        try:
            return f"{v:.3f}" if (v is not None and np.isfinite(v)) else "NA"
        except Exception:
            return "NA"

    # ---------- Load instance ----------
    rail_path = getattr(args, "rail_path", None)
    if not rail_path:
        # No rail_path: do NOT call an undefined synthetic runner here.
        # Return a benign row so the caller doesn't crash.
        return {
            "family":   "RAIL582",
            "dataset":  "rail582",
            "n":        np.nan,
            "m":        np.nan,
            "k":        int(getattr(args, "kcover", 1)),
            "seed":     int(getattr(args, "seed", 0)),
            "controller": str(getattr(args, "controller", "both")),
            "hyb_int":  np.nan,
            "mip_obj":  np.nan,
            "gap_pct":  np.nan,
            "hyb_total": 0.0,
            "polish_time": 0.0,
            "mip_time": np.nan,
            "relax_obj": np.nan,
            "round_obj": np.nan,
            "polish_obj": np.nan,
            "relax_time": 0.0,
            "round_main": 0.0,
            "round_rr": 0.0,
            "improve": 0.0,
            "feasible": 0,
            "note": "no --rail_path provided",
        }

    try:
        inst = load_orlib_rail(rail_path, k=getattr(args, "kcover", 1))
    except Exception as e:
        print(f"[rail] Failed to load '{rail_path}': {e}")
        return {"family": "RAIL582", "error": f"load failed: {e}"}

    # seed again after load for determinism
    if getattr(args, "seed", None) is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        _ensure_rows_of_col(inst)

    print(f"[rail] Loaded: n={inst.n} rows, m={inst.m} cols, k={inst.k}")

    # ---------- LP lower bound (optional) ----------
    lp_res = None
    if bool(getattr(args, "lp_lower_bound", False)):
        try:
            t0 = _pc()
            lp_res = solve_lp(inst, timelimit=getattr(args, "timelimit", None))
            _T["lp_solve"] += (_pc() - t0)
            if isinstance(lp_res, dict) and "obj" in lp_res:
                print(f"[rail] LP lower bound = {lp_res['obj']:.3f} (t={lp_res.get('time', float('nan')):.2f}s)")
        except Exception as e:
            print(f"[rail] LP failed: {e}")

    # ---------- MIP solve (if requested) ----------
# ---------- MIP solve (if requested) ----------
    mip_res = None
    ctrl = getattr(args, "controller", "both") or "both"
    if ctrl in ("mip", "both", "onego"):
        try:
            t0 = _pc()
            mip_res = solve_mip(
                inst, 
                timelimit=getattr(args, "timelimit", 60.0),
                gurobi_time_limit=getattr(args, "gurobi_time_limit", None),
                gurobi_gap_limit=getattr(args, "gurobi_gap_limit", None),
                track_anytime=True
            )
            _T["mip_solve"] += (_pc() - t0)
            
            # DEBUG: Check what's in mip_res
            print(f"[DEBUG] mip_res keys: {mip_res.keys() if isinstance(mip_res, dict) else 'not a dict'}")
            if isinstance(mip_res, dict) and 'anytime_log' in mip_res:
                print(f"[DEBUG] mip_res anytime_log has {len(mip_res['anytime_log'])} entries")
                if mip_res['anytime_log']:
                    print(f"[DEBUG] First entry: {mip_res['anytime_log'][0]}")
            
            # Merge GUROBI anytime data
            if isinstance(mip_res, dict) and 'anytime_log' in mip_res and mip_res['anytime_log']:
                for t, obj, method in mip_res['anytime_log']:
                    anytime_log.append((t, obj, f'mip_{method}'))
            
            if isinstance(mip_res, dict) and "obj" in mip_res:
                _g = mip_res.get("gap", None)
                _t = mip_res.get("time", None)
                print(f"[rail] MIP obj={_fmt(mip_res['obj'])}, "
                      f"gap={_fmt(_g)}, "
                      f"t={_fmt(_t)}s")
        except Exception as e:
            print(f"[rail] MIP failed: {e}")
            
    # If pure MIP, stop here (still return a row)
    if ctrl not in ("hybrid", "both", "onego"):
        T_total = _pc() - _tall0
        row = {
            "family":        "RAIL582",
            "dataset":       "rail582",
            "n":             int(inst.n),
            "m":             int(inst.m),
            "k":             int(inst.k),
            "seed":          int(getattr(args, "seed", 0)),
            "controller":    str(ctrl),
            "mip_obj":       float(mip_res["obj"]) if (mip_res and "obj" in mip_res) else np.nan,
            "mip_time":      float(mip_res.get("time", np.nan)) if mip_res else np.nan,
            "hyb_int":       np.nan,
            "gap_pct":       np.nan,
            "hyb_total":     float(T_total),
            "polish_time":   0.0,
            "relax_time":    0.0,
            "round_main":    0.0,
            "round_rr":      0.0,
            "improve":       0.0,
            "feasible":      0,
        }
        try:
            print(f"[rail] HYB skipped (controller={ctrl}); T={T_total:.2f}s")
        except Exception:
            pass
        return row
    
    # ---------- Tau schedule (optional) ----------
    tau_sched = None
    if getattr(args, "tau_schedule", None):
        try:
            tau_sched = [float(t) for t in args.tau_schedule.split(",")]
        except Exception:
            print("[rail] Could not parse --tau_schedule, ignoring.")

    # ---------- Relaxation ----------
    try:
        t0 = _pc()
        x_frac, y, obj_relax, t_relax, lin_relax, cov_min, cov_avg = entropy_relax_setcover_ipf_anneal(
            inst,
            tau=(getattr(args, "tau", None) if getattr(args, "tau", None) is not None else 0.2),
            iters=(getattr(args, "iters", None) if getattr(args, "iters", None) is not None else 50),
            tol=(getattr(args, "tol", None) if getattr(args, "tol", None) is not None else 1e-3),
            clip_one=(inst.k == 1),
            tau_schedule=tau_sched,
            extra_sweeps=120
        )
        _T["relax_total"] += (float(t_relax) if (t_relax is not None) else (_pc() - t0))
    except Exception as e:
        print(f"[rail] Relaxation failed: {e}")
        return {"family": "RAIL582", "error": f"relax failed: {e}"}

    # ---------- Anytime Performance Tracking ----------
    t_start = _pc()  # Start timing from here
    
    # After relaxation - no feasible solution yet
    anytime_log.append((t_relax, float('inf'), 'relaxation'))
    
    # ---------- Rounding & Improvement ----------
    feasible_int = False
    obj_hyb = float('inf')
    obj_hyb_pre = None
    x_int = None
    cov_min_int = None

    # Coalesce knobs to avoid None-type comparisons
    rr_trials = int(getattr(args, 'rr_trials', 0) or 0)
    rr_thr_max = float(getattr(args, 'rr_thr_max', 0.6) or 0.6)
    improve_passes = int(getattr(args, 'improve_passes', 0) or 0)
    rounding_mode = (getattr(args, 'rounding', 'dual') or 'dual')
    polish_time = float(getattr(args, 'polish_time', 3.0) or 3.0)
    polish_pool = float(getattr(args, 'polish_pool', 0.3) or 0.3)

    try:
        # randomized rounding (optional)
        if rr_trials > 0:
            t0 = _pc()
            xi, ci, covi, feasi = randomized_rounding_trials(
                inst, x_frac,
                trials=rr_trials,
                thr_max=rr_thr_max
            )
            t_rr = _pc() - t0
            _T["round_rr_trials"] += t_rr
            if feasi and (ci is not None) and np.isfinite(ci) and (ci < obj_hyb):
                x_int, obj_hyb, cov_min_int, feasible_int = xi, float(ci), covi, True
                # Log anytime improvement
                anytime_log.append((t_relax + t_rr, obj_hyb, 'randomized_rounding'))

        # main rounding
        if not feasible_int:
            t0 = _pc()
            if rounding_mode == "dual":
                xi, ci, covi, feasi = round_cover_dual(inst, x_frac, y)
            else:
                xi, ci, covi, feasi = round_cover_greedy(inst, x_frac)
            t_round = _pc() - t0
            _T["round_main"] += t_round
            if feasi and (ci is not None) and np.isfinite(ci):
                x_int, obj_hyb, cov_min_int, feasible_int = xi, float(ci), covi, True
                # Log anytime improvement
                total_time = t_relax + (_T["round_rr_trials"] if rr_trials > 0 else 0) + t_round
                anytime_log.append((total_time, obj_hyb, 'main_rounding'))

        # keep pre-polish objective for reporting
        obj_hyb_pre = (obj_hyb if (np.isfinite(obj_hyb)) else None)

        # local improve
        if feasible_int and (improve_passes > 0):
            t0 = _pc()
            xi, ci, covi, feasi = local_improve_drop_and_fix(
                inst, x_int, passes=improve_passes
            )
            t_improve = _pc() - t0
            _T["improve_local"] += t_improve
            if feasi and (ci is not None) and np.isfinite(ci) and (ci < obj_hyb):
                x_int, obj_hyb, cov_min_int, feasible_int = xi, float(ci), covi, True
                # Log anytime improvement
                total_time = t_relax + (_T["round_rr_trials"] if rr_trials > 0 else 0) + _T["round_main"] + t_improve
                anytime_log.append((total_time, obj_hyb, 'local_improvement'))

    except Exception as e:
        print(f"[rail] Rounding/improvement failed: {e}")
        feasible_int = False
    
    # DEBUG: Check polish parameters and state before polish
    if feasible_int and x_int is not None:
        print(f"[DEBUG POLISH] polish_time={getattr(args, 'polish_time', None)}, "
              f"polish_pool={getattr(args, 'polish_pool', None)}, "
              f"x_int nonzero={np.sum(x_int)}, "
              f"y is None={y is None}")        

    # ---------- Polish (restricted MIP) ----------
    hyb_polished = False
    try:
        if feasible_int and (x_int is not None) and (y is not None):
            # time reduced costs explicitly
            t0 = _pc()
            r = compute_reduced_costs(inst, y)
            _T["reduced_costs"] += (_pc() - t0)

            # use CLI-provided values directly
            t0 = _pc()
            try:
                x_pol, cost_pol, feas_pol, t_pol = polish_restricted_mip(
                    inst, x_init=x_int, r=r,
                    timelimit=polish_time,
                    pool_frac=polish_pool,
                )
            except Exception:
                x_pol, cost_pol, feas_pol, t_pol = polish_restricted_mip_pulp(
                    inst, x_init=x_int, r=r,
                    timelimit=polish_time,
                    pool_frac=polish_pool,
                )

            # count polish time once (prefer routine's own timing)
            _T["polish_mip"] += float(t_pol) if (t_pol is not None) else 0.0

            if feas_pol and (cost_pol is not None) and np.isfinite(cost_pol) and (cost_pol < obj_hyb):
                x_int, obj_hyb = x_pol, float(cost_pol)
                hyb_polished = True
                # Log final anytime improvement
                total_time = (t_relax + _T["round_rr_trials"] + _T["round_main"] + 
                             _T["improve_local"] + _T["polish_mip"])
                anytime_log.append((total_time, obj_hyb, 'polish'))
                
    except Exception as e:
        print(f"[rail] Polish failed (continuing): {e}")

    obj_hyb_post = (obj_hyb if np.isfinite(obj_hyb) else obj_hyb_pre)
    T_total = _pc() - _tall0
    sum_stages = sum(_T.values())

    # ---------- Console summary ----------
    hyb_line = (f"[rail] HYB int={_fmt(obj_hyb_post)}"
                f"{' +polish' if hyb_polished else ''} "
                f"(feas={'OK' if feasible_int else 'INFEAS'}), "
                f"relax_lin={_fmt(lin_relax)}, cov_min_frac={_fmt(cov_min)}, "
                f"cov_min_int={_fmt(cov_min_int)}, "
                f"T={T_total:.2f}s (relax={_fmt(t_relax)}s, Sum_stages={sum_stages:.2f}s) "
                f"| objs: relax={_fmt(obj_relax)}, round={_fmt(obj_hyb_pre)}, polish={_fmt(obj_hyb_post)}")
    
    if isinstance(lp_res, dict) and ('obj' in lp_res):
        hyb_line += f", lp={_fmt(lp_res['obj'])}"
    if isinstance(mip_res, dict) and ('obj' in mip_res):
        hyb_line += f", mip={_fmt(mip_res['obj'])}"
        if feasible_int and (obj_hyb_post is not None) and np.isfinite(obj_hyb_post):
            pct_gap = (obj_hyb_post - mip_res['obj']) / max(1e-9, abs(mip_res['obj']))
            hyb_line += f", gap%={100.0*pct_gap:.2f}"

    hyb_line += (f" | times(s): relax={_T['relax_total']:.2f}, "
                 f"round_main={_T['round_main']:.2f}, "
                 f"round_rr={_T['round_rr_trials']:.2f}, "
                 f"improve={_T['improve_local']:.2f}, "
                 f"polish={_T['polish_mip']:.2f}, "
                 f"mip={_T['mip_solve']:.2f}, "
                 f"lp={_T['lp_solve']:.2f}, "
                 f"redcost={_T['reduced_costs']:.2f})")
    try:
        print(hyb_line)
    except Exception:
        pass

    # ---------- Build single summary row ----------
    gap_pct = np.nan
    if mip_res and isinstance(mip_res, dict) and "obj" in mip_res and np.isfinite(obj_hyb_post or np.nan):
        denom = max(1e-9, abs(float(mip_res["obj"])))
        gap_pct = 100.0 * (float(obj_hyb_post) - float(mip_res["obj"])) / denom

    row = {
        "family":        "RAIL582",
        "dataset":       "rail582",
        "n":             int(inst.n),
        "m":             int(inst.m),
        "k":             int(inst.k),
        "seed":          int(getattr(args, "seed", 0)),
        "controller":    str(ctrl),
        "hyb_int":       float(obj_hyb_post) if (obj_hyb_post is not None and np.isfinite(obj_hyb_post)) else np.nan,
        "mip_obj":       float(mip_res["obj"]) if (mip_res and "obj" in mip_res) else np.nan,
        "gap_pct":       float(gap_pct) if np.isfinite(gap_pct) else np.nan,
        "hyb_total":     float(T_total),
        "polish_time":   float(_T["polish_mip"]),
        "mip_time":      float(mip_res.get("time", np.nan)) if mip_res else np.nan,
        "relax_obj":     float(obj_relax) if (obj_relax is not None and np.isfinite(obj_relax)) else np.nan,
        "round_obj":     float(obj_hyb_pre) if (obj_hyb_pre is not None and np.isfinite(obj_hyb_pre)) else np.nan,
        "polish_obj":    float(obj_hyb_post) if (obj_hyb_post is not None and np.isfinite(obj_hyb_post)) else np.nan,
        "relax_time":    float(_T["relax_total"]),
        "round_main":    float(_T["round_main"]),
        "round_rr":      float(_T["round_rr_trials"]),
        "improve":       float(_T["improve_local"]),
        "feasible":      int(bool(feasible_int)),
    }
    
    # Add anytime performance data
    if anytime_log:
        for i, (t, obj, method) in enumerate(anytime_log):
            row[f"anytime_t_{i}"] = float(t)
            row[f"anytime_obj_{i}"] = float(obj) if np.isfinite(obj) else np.nan
            row[f"anytime_method_{i}"] = str(method)
    
        # Also add a compact anytime summary
        anytime_summary = "|".join([f"{method}:{obj:.0f}@{t:.3f}s" 
                                   for t, obj, method in anytime_log 
                                   if np.isfinite(obj)])
        row["anytime_summary"] = anytime_summary
    
    return row

def run_family_synthetic(
    scales,
    trials=3,
    tau=0.2,
    timelimit=60.0,
    seed=2025,
    args_k=1,
    controller='both',
    iters=50,
    tol=1e-3,
    relax_engine='ipf',
    tau_schedule=None,
    rounding='dual',
    lp_lower_bound: bool = True,
    # NEW: let synthetic runs use the same polish knobs (can be None to use shared defaults)
    polish_time=None,
    polish_pool=None,
):
    """
    Synthetic set-cover benchmark (HYB + MIP). Safe against None/inf comparisons,
    supports CLI/central polish controls, and avoids backend gating for polish.
    """
    import numpy as np

    # Coalesce polish defaults from central shared defaults if provided
    if (polish_time is None) or (polish_pool is None):
        try:
            from mip_hybrid.shared_defaults import HYBRID_DEFAULTS
            if polish_time is None:
                polish_time = HYBRID_DEFAULTS.get("polish_time", 3.0)
            if polish_pool is None:
                polish_pool = HYBRID_DEFAULTS.get("polish_pool", 0.3)
        except Exception:
            polish_time = 3.0 if polish_time is None else polish_time
            polish_pool = 0.3 if polish_pool is None else polish_pool

    rows = []
    for (n, m) in scales:
        for rep in range(trials):
            seed_used = int(seed + 1000 * rep + n + m)
            inst = make_set_cover(n, m, seed=seed_used, k=args_k)

            lp = solve_lp(inst, timelimit=timelimit) if lp_lower_bound else None
            mip = solve_mip(inst, timelimit=timelimit) if controller in ('mip', 'both') else None

            # Initialize HYB outputs to safe sentinels up front
            x_frac = y = obj_relax = t_relax = lin_relax = cov_min = cov_avg = None
            x_int = None
            obj_hyb = float('inf')
            cov_min_int = None
            feasible_int = False
            t_hyb_total = None
            hyb_polished = False

            if controller in ('hybrid', 'both'):
                taus = [float(t) for t in tau_schedule.split(",")] if tau_schedule else None
                x_frac, y, obj_relax, t_relax, lin_relax, cov_min, cov_avg = entropy_relax_setcover_ipf_anneal(
                    inst, tau=tau, iters=iters, tol=tol,
                    clip_one=(inst.k == 1),
                    tau_schedule=taus, extra_sweeps=120
                )

                # Rounding
                if rounding == 'dual':
                    xi, ci, covi, feasi = round_cover_dual(inst, x_frac, y)
                else:
                    xi, ci, covi, feasi = round_cover_greedy(inst, x_frac)

                # Accept only sane results
                if feasi and (ci is not None) and np.isfinite(ci):
                    x_int, obj_hyb, cov_min_int, feasible_int = xi, float(ci), covi, True

                # Base HYB time starts from relaxation time
                t_hyb_total = float(t_relax) if (t_relax is not None) else 0.0

                # ---------- Polish (restricted MIP) ----------
                # Run polish only if we truly have an incumbent and duals
                if (y is not None) and (x_int is not None) and feasible_int:
                    try:
                        r = compute_reduced_costs(inst, y)
                        try:
                            x_pol, cost_pol, feas_pol, t_pol = polish_restricted_mip(
                                inst, x_init=x_int, r=r,
                                timelimit=polish_time,
                                pool_frac=polish_pool,
                            )
                        except Exception:
                            x_pol, cost_pol, feas_pol, t_pol = polish_restricted_mip_pulp(
                                inst, x_init=x_int, r=r,
                                timelimit=polish_time,
                                pool_frac=polish_pool,
                            )

                        # Accumulate polish time once
                        if t_pol is None:
                            t_pol = 0.0
                        t_hyb_total += float(t_pol)

                        # Accept only a real improvement
                        if feas_pol and (cost_pol is not None) and np.isfinite(cost_pol) and (cost_pol < obj_hyb):
                            x_int, obj_hyb = x_pol, float(cost_pol)
                            # Recompute integer coverage to confirm feasibility
                            cover_cnt = np.zeros(inst.n, dtype=int)
                            for i in range(inst.n):
                                cover_cnt[i] = sum(x_int[j] for j in inst.A[i])
                            cov_min_int = float(cover_cnt.min())
                            feasible_int = bool(cov_min_int >= inst.k)
                            hyb_polished = True
                    except Exception as e:
                        print(f"[syn] Polish failed (continuing): {e}")

            else:
                # pure MIP path: keep HYB fields as initialized above
                pass

            # ---------- Gaps ----------
            pct_gap = None
            if (mip is not None and mip.get('gap') is not None
                and abs(mip.get('gap')) < 1e-9
                and feasible_int
                and (obj_hyb is not None) and np.isfinite(obj_hyb)):
                denom = max(1e-9, float(mip['obj']))
                pct_gap = (float(obj_hyb) - float(mip['obj'])) / denom

            lp_gap_hyb = None
            if lp_lower_bound and (lp is not None) and feasible_int and (obj_hyb is not None) and np.isfinite(obj_hyb):
                denom_lp = max(1e-9, float(lp['obj']))
                lp_gap_hyb = (float(obj_hyb) - float(lp['obj'])) / denom_lp

            lp_gap_mip = None
            if lp_lower_bound and (lp is not None) and (mip is not None):
                denom_lp = max(1e-9, float(lp['obj']))
                lp_gap_mip = (float(mip['obj']) - float(lp['obj'])) / denom_lp

            # ---------- Row ----------
            rows.append({
                "family": "setcover",
                "n": n, "m": m, "tau": tau, "trial": rep + 1,
                "k": args_k,
                "seed_used": seed_used,
                "controller": controller,
                "relax_engine": relax_engine,
                "rounding": rounding,
                "tau_schedule": tau_schedule,
                "lp_backend": (lp.get('backend') if (lp_lower_bound and lp is not None) else None),
                "lp_obj": (lp['obj'] if (lp_lower_bound and lp is not None) else None),
                "lp_time": (lp['time'] if (lp_lower_bound and lp is not None) else None),
                "mip_backend": (mip.get('backend') if mip is not None else None),
                "mip_obj": (mip['obj'] if mip is not None else None),
                "mip_gap": (mip.get('gap') if mip is not None else None),
                "mip_time": (mip['time'] if mip is not None else None),
                "mip_status": (mip['status'] if mip is not None else None),
                "hyb_relax_obj": obj_relax,
                "hyb_relax_lin_cost": lin_relax,
                "hyb_relax_cov_min": cov_min,
                "hyb_relax_cov_avg": cov_avg,
                "hyb_relax_time": t_relax,
                "hyb_int_obj": (float(obj_hyb) if np.isfinite(obj_hyb) else None),
                "hyb_int_cov_min": cov_min_int,
                "hyb_int_feasible": bool(feasible_int),
                "hyb_total_time": t_hyb_total,
                "hyb_pct_gap_vs_mip": (float(pct_gap) if pct_gap is not None else None),
                "hyb_vs_lp_pct_gap": (float(lp_gap_hyb) if lp_gap_hyb is not None else None),
                "mip_vs_lp_pct_gap": (float(lp_gap_mip) if lp_gap_mip is not None else None),
                "hyb_polished": bool(hyb_polished),
                "hyb_polish_time": (
                    (float(t_hyb_total) - float(t_relax))
                    if (t_hyb_total is not None and t_relax is not None and hyb_polished)
                    else 0.0
                ),
            })

            # ---------- Console summary ----------
            if controller == 'both' and (mip is not None) and np.isfinite(obj_hyb):
                feas = "OK" if feasible_int else "INFEAS"
                gap_str = f"{(pct_gap * 100):.2f}%" if (pct_gap is not None) else "NA"
                pol = " +polish" if hyb_polished else ""
                print(
                    f"[n={n}, m={m}, rep={rep+1}]  "
                    f"MIP obj={mip['obj']:.3f}, gap={(mip['gap'] if mip.get('gap') is not None else float('nan')):.3f}, t={mip['time']:.2f}s  |  "
                    f"HYB({relax_engine},{rounding}) int={obj_hyb:.3f}{pol} ({feas}, gap%={gap_str}), "
                    f"lin-relax={(lin_relax if lin_relax is not None else float('nan')):.3f}, "
                    f"cov_min_frac={(cov_min if cov_min is not None else float('nan')):.2f}, "
                    f"cov_min_int={(cov_min_int if cov_min_int is not None else float('nan')):.2f}, "
                    f"t={(t_hyb_total if t_hyb_total is not None else float('nan')):.2f}s"
                )
            elif controller == 'mip' and mip is not None:
                print(f"[n={n}, m={m}, rep={rep+1}]  MIP obj={mip['obj']:.3f}, gap={(mip['gap'] if mip.get('gap') is not None else float('nan')):.3f}, t={mip['time']:.2f}s")
            elif controller == 'hybrid' and np.isfinite(obj_hyb):
                feas = "OK" if feasible_int else "INFEAS"
                pol = " +polish" if hyb_polished else ""
                print(
                    f"[n={n}, m={m}, rep={rep+1}]  "
                    f"HYB({relax_engine},{rounding}) int={obj_hyb:.3f}{pol} ({feas}), "
                    f"lin-relax={(lin_relax if lin_relax is not None else float('nan')):.3f}, "
                    f"cov_min_frac={(cov_min if cov_min is not None else float('nan')):.2f}, "
                    f"cov_min_int={(cov_min_int if cov_min_int is not None else float('nan')):.2f}, "
                    f"t={(t_hyb_total if t_hyb_total is not None else float('nan')):.2f}s"
                )

    return rows

# In mip_hybrid/apps/rail_setcover.py
# Replace the existing write_csv function with this:

def write_csv(rows: List[dict], out_path: Optional[str] = None):
    """Write results to CSV with dynamic column handling"""
    if not rows or out_path is None: 
        return
    
    # Collect all possible field names from all rows
    all_fieldnames = set()
    for row in rows:
        all_fieldnames.update(row.keys())
    
    # Sort for consistent column ordering
    fieldnames = sorted(list(all_fieldnames))
    
    print(f"[rail] Writing {len(rows)} rows with {len(fieldnames)} columns to {out_path}")
    
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    
    print(f"[rail] Successfully wrote {out_path}")

def _ensure_rows_of_col(inst):
    """Build inst._rows_of_col (rows per column) once, cached on the instance."""
    if getattr(inst, "_rows_of_col", None) is not None:
        return

    rows_of_col = [[] for _ in range(inst.m)]
    for i, cols in enumerate(inst.A):
        for j in cols:
            rows_of_col[j].append(i)

    inst._rows_of_col = [
        np.asarray(r, dtype=np.int32) if r else np.empty(0, np.int32)
        for r in rows_of_col
    ]


def parse_scales(s: str):
    out = []
    for token in s.split(","):
        n,m = token.split("x")
        out.append((int(n), int(m)))
    return out

def _run(args):
    """
    Shared core used by both: python -m (via main()) and cli.py (via in-process call).
    Expects args to have:
      rail_path, out, out_dir?, trials, seed, solver,
      controller, profile, profile_out,
      kcover, tau_schedule, tol,
      polish_time, polish_pool, rr_trials, timelimit, rounding, etc.
    """
    
    # --- global threading controls (NumPy/BLAS + GUROBI) ---
    import os as __os_env
    thr = int(getattr(args, "threads", 1) or 1)
    gap = float(getattr(args, "mip_gap", 0.01) or 0.01)

    # Force single-threaded math libraries unless user overrides
    for k in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:
        __os_env.environ[k] = str(thr)

    # For GUROBI via PuLP (GUROBI_CMD), stick to env
    __os_env.environ["GRB_THREADS"] = str(thr)
    __os_env.environ["THREADS"] = str(thr)
    __os_env.environ["GRB_MIPGAP"] = str(gap)
    __os_env.environ["MIP_GAP"] = str(gap)
# --- solver selection from CLI / env ---
    global _SELECTED_SOLVER
    _os.environ["MIP_SOLVER"] = args.solver   # if other layers want to read it
    _SELECTED_SOLVER = args.solver.lower()

    # keep a module-level selected solver for this app run
    global _SOLVER
    _SOLVER = args.solver  # "cbc" or "gurobi"

    def backend_name():
        # report the user-selected backend, not the detected python package
        return _SOLVER.upper()

    # auto-enable profiling in onego
    if args.controller == "onego":
        args.profile = True
        if not args.profile_out:
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            args.profile_out = f"profile_{ts}.json"

    # light schedule tweaks for k-cover
    if args.kcover >= 2 and (args.tau_schedule is None):
        args.tau_schedule = "0.5,0.2,0.1,0.05,0.02"
    if args.kcover >= 2 and args.tol == 1e-3:
        args.tol = 5e-4

    # resolve rail path based on RUN_MODE
    mode = RUN_MODE.lower()
    rail_auto = resolve_rail_path(args.rail_path)
    if mode == "rail":
        args.rail_path = rail_auto
    elif mode == "synthetic":
        args.rail_path = None
    else:
        args.rail_path = rail_auto

    print(f"[mode] RUN_MODE={RUN_MODE} | rail_path={repr(args.rail_path)} | threads={getattr(args, 'threads', 1)} | mip_gap={getattr(args, 'mip_gap', 0.01)}")

    # ----- RAIL branch -----
    if args.rail_path:
        rows = []
        base_seed = int(getattr(args, "seed", 0) or 0)
        trials = int(getattr(args, "trials", 1) or 1)

        for t in range(trials):
            args.seed = base_seed + t
            row = run_family(args)   # <-- uses args.polish_time / args.polish_pool downstream
            if isinstance(row, dict) and row:
                row.setdefault("trial", t)
                rows.append(row)

        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = args.out or f"rail582_results_{ts}.csv"
        if rows:
            write_csv(rows, out_path)
            print(f"[rail] wrote {out_path} rows={len(rows)}")
        else:
            print("[rail] no rows to write (check log)")
        return

    # ----- synthetic branch -----
    rows = _run_family_synthetic(args)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M")
    out = args.out or f"setcover_benchmark_pub_{ts}.csv"
    write_csv(rows, out)
    print("Wrote results →", out)
    print("MIP backend:", selected_solver().upper())


def main():
    p = argparse.ArgumentParser(description="Set Cover benchmark: MIP vs Entropy Hybrid (optimized+profiled)")
    # --- synthetic experiment args ---
    p.add_argument("--scales", type=str, default="400x200,800x400,1600x800")
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--tau_schedule", type=str, default=None)
    p.add_argument("--timelimit", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--kcover", type=int, default=1)
    p.add_argument("--controller", type=str, default="both", choices=["mip","hybrid","both","onego"])
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--rounding", type=str, default="dual", choices=["dual","greedy"])
    p.add_argument("--lp_lower_bound", action="store_true")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--solver",
                   choices=["cbc", "gurobi"],
                   default="cbc",
                   help="MIP solver backend (default: cbc)")

    # --- profiling ---
    p.add_argument("--profile", action="store_true", help="Enable fine-grained timing")
    p.add_argument("--profile_out", type=str, default=None, help="Write profile JSON")

    # --- RAIL args ---
    p.add_argument("--rail_path", type=str, default=None)
    p.add_argument("--rr_trials", type=int, default=1)
    p.add_argument("--rr_thr_max", type=float, default=0.8)
    p.add_argument("--improve_passes", type=int, default=1)
    p.add_argument("--polish_time", type=float, default=3.0)
    p.add_argument("--polish_pool", type=float, default=0.25)
    p.add_argument("--threads", type=int, default=1, help="Total compute threads (BLAS/MIP). Default=1")
    p.add_argument("--mip_gap", type=float, default=0.01, help="Relative MIP gap for GUROBI (e.g., 0.01)")
    
    

    args = p.parse_args()
    _run(args)

def main_from_cli_namespace(
    rail_path,
    out=None,
    out_dir=None,             # not used here, but kept for symmetry with CLI
    trials=2,
    solver="cbc",
    # --- keep these defaults in sync with your argparse above ---
    scales="400x200,800x400,1600x800",
    tau=0.1,
    tau_schedule=None,
    timelimit=60.0,
    seed=2025,
    kcover=1,
    controller="both",        # choices=["mip","hybrid","both","onego"]
    iters=200,
    tol=1e-3,
    rounding="dual",          # choices=["dual","greedy"]
    lp_lower_bound=False,
    profile=False,
    profile_out=None,
    rr_trials=1,
    rr_thr_max=0.8,
    improve_passes=1,
    polish_time=3.0,
    polish_pool=0.25,
    threads=1,
    mip_gap=0.01,         # set to 0.30 here if you prefer that default
    gurobi_time_limit=None,      # <-- Use None as default
    gurobi_gap_limit=None,       # <-- Use None as default  
    track_gurobi_anytime=False,  # <-- Use False as default
):
    """
    In-process entrypoint for runners/cli to call this app without a subprocess.
    Builds an args-like namespace and forwards to _run(args).
    """
    print(f"[DEBUG] main_from_cli_namespace called with: gurobi_time_limit={gurobi_time_limit}, gurobi_gap_limit={gurobi_gap_limit}, track_gurobi_anytime={track_gurobi_anytime}")
    args = SimpleNamespace(
        # outputs & dataset
        rail_path=rail_path,
        out=out,
        out_dir=out_dir,   # harmless if unused inside this module

        # generic controls (mirror your argparse)
        scales=scales,
        trials=trials,
        tau=tau,
        tau_schedule=tau_schedule,
        timelimit=timelimit,
        seed=seed,
        kcover=kcover,
        controller=controller,
        iters=iters,
        tol=tol,
        rounding=rounding,
        lp_lower_bound=lp_lower_bound,
        solver=solver,

        # profiling
        profile=profile,
        profile_out=profile_out,

        # rail-specific / hybrid knobs
        rr_trials=rr_trials,
        rr_thr_max=rr_thr_max,
        improve_passes=improve_passes,
        polish_time=polish_time,
        polish_pool=polish_pool,
        threads=threads,
        mip_gap=mip_gap,
        gurobi_time_limit=gurobi_time_limit,
        gurobi_gap_limit=gurobi_gap_limit,
        track_gurobi_anytime=track_gurobi_anytime,
    )
    _run(args)
