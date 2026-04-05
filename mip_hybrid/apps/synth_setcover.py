# mip_hybrid/apps/synth_setcover.py - Fixed version using entropy framework
import argparse, os, time
import numpy as np
import pandas as pd
import os as _os
from dataclasses import dataclass
from typing import List, Optional
import math

# --- selected solver (cbc | gurobi) controlled by CLI/env ---
_SELECTED_SOLVER = _os.getenv("MIP_SOLVER", "gurobi").lower()

def selected_solver():
    return "gurobi"  # Force Gurobi for anytime tracking

@dataclass
class SetCoverInstance:
    A: List[List[int]]   # element -> list of sets covering it
    c: np.ndarray        # costs (m,)
    n: int               # number of elements
    m: int               # number of sets
    k: int = 1           # coverage requirement (>=k)
    # Fast incidence views
    _rows_of_col: Optional[List[np.ndarray]] = None

def _parse_scales(s):
    items = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok: continue
        n, m = tok.lower().split("x")
        items.append((int(n), int(m)))
    return items

def _gen_entropy_friendly_scp(n, m, seed=42):
    """Generate instances designed to showcase entropy-based methods"""
    rng = np.random.default_rng(seed)
    
    # Higher density for complex structure
    density = 0.02  # Much denser than current 0.002
    A = (rng.random((n, m)) < density).astype(np.uint8)
    
    # Ensure every row is covered (required for feasibility)
    row_cov = A.sum(axis=1)
    for i, c in enumerate(row_cov):
        if c == 0:
            j = rng.integers(0, m)
            A[i, j] = 1
    
    # Cost structure that benefits from dual information
    # Make costs correlated with coverage (more coverage = higher cost)
    col_coverage = A.sum(axis=0)
    base_costs = rng.random(m) * 0.5 + 0.1  # Base random cost
    coverage_penalty = col_coverage / col_coverage.max() * 0.5  # Higher coverage = higher cost
    c = base_costs + coverage_penalty
    
    return A, c

def _build_instance_from_matrix(A_matrix, c):
    """Convert matrix format to instance format used by entropy solver"""
    n, m = A_matrix.shape
    # Build A as list of lists: A[i] = [columns that cover row i]
    A = []
    for i in range(n):
        cols = np.where(A_matrix[i, :] == 1)[0].tolist()
        A.append(cols)
    
    inst = SetCoverInstance(A=A, c=c, n=n, m=m, k=1)
    
    # Build _rows_of_col: for each column j, which rows does it cover
    rows_of_col = [[] for _ in range(m)]
    for i, cols in enumerate(A):
        for j in cols:
            rows_of_col[j].append(i)
    
    inst._rows_of_col = [
        np.array(r, dtype=np.int32) if r else np.empty(0, dtype=np.int32)
        for r in rows_of_col
    ]
    
    return inst

# ======= Port the entropy framework from rail_setcover.py =======

def ipf_rowwise_entropy(inst: SetCoverInstance, tau: float, iters: int, tol: float):
    """
    Row-wise IPF entropy relaxation - simplified version from rail_setcover.py
    """
    t0 = time.time()
    q = np.exp(-inst.c / max(tau, 1e-12))
    x = q.copy()
    y = np.zeros(inst.n, dtype=float)
    
    rhs = float(inst.k)
    
    for sweep in range(iters):
        max_violation = 0.0
        for i in range(inst.n):
            cols = inst.A[i]
            if not cols:
                continue
            cols = np.array(cols)
            s = float(x[cols].sum())
            if s < rhs - tol:
                alpha = rhs / max(s, 1e-12)
                x[cols] *= alpha
                y[i] += tau * math.log(alpha)
                max_violation = max(max_violation, rhs - s)
        
        # Clip to [0,1] for set cover
        np.minimum(x, 1.0, out=x)
        
        if max_violation <= tol:
            break
    
    # Compute coverage and objectives
    cov = np.zeros(inst.n, dtype=float)
    for i in range(inst.n):
        cov[i] = x[inst.A[i]].sum() if inst.A[i] else 0.0
    
    lin_cost = float(inst.c @ x)
    smooth_obj = lin_cost + tau * float(np.sum(x * (np.log(np.maximum(x,1e-12)) - 1.0)))
    
    return x, y, smooth_obj, time.time() - t0, lin_cost, float(cov.min()), float(cov.mean())

def entropy_relax_with_annealing(inst: SetCoverInstance, tau: float = 0.1, 
                                iters: int = 50, tol: float = 1e-3,
                                tau_schedule: Optional[List[float]] = None):
    """
    Run IPF with tau annealing - simplified from rail_setcover.py
    """
    taus = tau_schedule if tau_schedule else [tau]
    x = None
    total_time = 0.0
    prev_t = tau
    
    print(f"[DEBUG annealing] Running {len(taus)} temperature stages: {taus}")
    
    for stage_idx, t in enumerate(taus):
        print(f"[DEBUG annealing] Stage {stage_idx+1}/{len(taus)}: tau={t}, iters={iters}")
        
        if x is not None:
            # Reweight for new tau
            reweight = np.exp(-(1.0/max(t,1e-12) - 1.0/max(prev_t,1e-12)) * inst.c)
            x = np.clip(x * reweight, 1e-18, 1.0)
            print(f"[DEBUG annealing] Reweighted x: min={x.min():.6f}, max={x.max():.6f}")
        
        stage_start = time.time()
        x, y, smooth_obj, dt, lin_cost, cov_min, cov_avg = ipf_rowwise_entropy(
            inst, tau=t, iters=iters, tol=tol
        )
        stage_time = time.time() - stage_start
        total_time += stage_time
        
        print(f"[DEBUG annealing] Stage {stage_idx+1} done: t={stage_time:.3f}s, lin_cost={lin_cost:.2f}, cov_min={cov_min:.3f}")
        
        prev_t = t
    
    print(f"[DEBUG annealing] Total annealing time: {total_time:.3f}s")
    return x, y, smooth_obj, total_time, lin_cost, cov_min, cov_avg

def compute_reduced_costs(inst: SetCoverInstance, y: np.ndarray):
    """Compute reduced costs r_j = c_j - sum_i y_i for i in rows covered by j"""
    r = inst.c.copy()
    for j in range(inst.m):
        rows = inst._rows_of_col[j]
        if rows.size > 0:
            r[j] -= y[rows].sum()
    return r


def round_cover_dual_guided(inst: SetCoverInstance, x_frac: np.ndarray, y: np.ndarray):
    """
    Dual-guided rounding - simplified from rail_setcover.py
    """
    n, m, k = inst.n, inst.m, inst.k
    r = compute_reduced_costs(inst, y)
    
    x_int = np.zeros(m, dtype=int)
    cover_cnt = np.zeros(n, dtype=int)
    
    # Take all non-positive reduced cost columns
    nz = np.where(r <= 1e-12)[0]
    if nz.size:
        x_int[nz] = 1
        for j in nz:
            rows = inst._rows_of_col[j]
            if rows.size:
                cover_cnt[rows] += 1
    
    # Greedy completion using reduced costs
    while np.any(cover_cnt < k):
        best_j = -1
        best_ratio = float('inf')
        
        # Find uncovered rows
        deficit_rows = np.where(cover_cnt < k)[0]
        candidates = set()
        for i in deficit_rows:
            for j in inst.A[i]:
                if x_int[j] == 0:
                    candidates.add(j)
        
        # Pick best candidate by reduced cost
        for j in candidates:
            rows = inst._rows_of_col[j]
            gain = np.sum(cover_cnt[rows] < k) if rows.size else 0
            if gain > 0:
                ratio = max(1e-12, r[j]) / gain
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_j = j
        
        if best_j == -1:
            # Fallback: pick cheapest column covering some deficit row
            i0 = deficit_rows[0]
            candidates = [j for j in inst.A[i0] if x_int[j] == 0]
            if not candidates:
                break
            best_j = min(candidates, key=lambda j: r[j])
        
        x_int[best_j] = 1
        rows = inst._rows_of_col[best_j]
        if rows.size:
            cover_cnt[rows] += 1
    
    # Simple pruning
    on_cols = np.where(x_int == 1)[0]
    for j in on_cols:
        rows = inst._rows_of_col[j]
        if rows.size == 0:
            x_int[j] = 0
            continue
        # Check if removing j violates any row
        if not np.any(cover_cnt[rows] - 1 < k):
            x_int[j] = 0
            cover_cnt[rows] -= 1
    
    cov_min = float(cover_cnt.min())
    feasible = bool(cov_min >= k)
    cost = float(inst.c @ x_int)
    
    return x_int, cost, cov_min, feasible

def solve_entropy_setcover(A_matrix, c, tau=0.1, iters=50, tol=1e-3, tau_schedule=None,
                          polish_time=1.0, polish_pool=0.3):
    """
    Main entropy solver with optional polish refinement.
    """
    import time
    t_start = time.time()
    
    inst = _build_instance_from_matrix(A_matrix, c)
    
    # ADD DIAGNOSTIC
    print(f"[DEBUG solve_entropy] Instance: {inst.n}x{inst.m}")
    print(f"[DEBUG solve_entropy] tau_schedule={tau_schedule}")
    print(f"[DEBUG solve_entropy] iters={iters}, tol={tol}")
    
    # Parse tau schedule if provided
    taus = None
    if tau_schedule:
        try:
            taus = [float(t.strip()) for t in tau_schedule.split(",")]
            print(f"[DEBUG solve_entropy] Parsed taus: {taus}")
        except Exception as e:
            print(f"[DEBUG solve_entropy] Failed to parse tau_schedule: {e}")
            taus = None
    
    # Entropy relaxation with annealing
    print(f"[DEBUG solve_entropy] Starting relaxation...")
    x_frac, y, smooth_obj, t_relax, lin_cost, cov_min, cov_avg = entropy_relax_with_annealing(
        inst, tau=tau, iters=iters, tol=tol, tau_schedule=taus
    )
    print(f"[DEBUG solve_entropy] Relaxation done: t={t_relax:.3f}s, lin_cost={lin_cost:.2f}, cov_min={cov_min:.2f}")
    
    # Dual-guided rounding
    print(f"[DEBUG solve_entropy] Starting rounding...")
    t_round_start = time.time()
    x_int, cost, cov_min_int, feasible = round_cover_dual_guided(inst, x_frac, y)
    t_round = time.time() - t_round_start
    print(f"[DEBUG solve_entropy] Rounding done: t={t_round:.3f}s, cost={cost:.2f}, feasible={feasible}")
    
    # Polish if requested
    total_time = t_relax + t_round
    if polish_time > 0 and feasible:
        print(f"[DEBUG solve_entropy] Starting polish...")
        x_int, cost, t_polish = polish_solution(
            x_int, A_matrix, c, 
            polish_time=polish_time, 
            polish_pool=polish_pool,
            x_frac=x_frac
        )
        total_time += t_polish
        print(f"[DEBUG solve_entropy] Polish done: t={t_polish:.3f}s, final_cost={cost:.2f}")
    
    print(f"[DEBUG solve_entropy] Total time: {total_time:.3f}s")
    return x_int, cost, feasible, total_time



def polish_solution(x_int, A, c, polish_time=1.0, polish_pool=0.3, x_frac=None):
    """
    Polish an integer solution using restricted MIP.
    
    Args:
        x_int: Current integer solution (numpy array)
        A: Constraint matrix (n x m)
        c: Cost vector (m,)
        polish_time: Time limit for Gurobi (seconds)
        polish_pool: Fraction of variables to include (0-1)
        x_frac: Optional fractional solution to guide variable selection
    
    Returns:
        (x_polished, cost_polished, time_taken)
    """
    import time
    import numpy as np
    
    t0 = time.time()
    
    n, m = A.shape
    
    # Select variables to include in polish
    # Strategy: include variables that are "on" in current solution
    # plus variables with fractional values close to 0.5 (most uncertain)
    if x_frac is not None:
        # Distance from 0.5 (closer = more uncertain)
        proximity = 1.0 - 2.0 * np.abs(x_frac - 0.5)
        top_k = int(m * polish_pool)
        candidates = np.argsort(-proximity)[:top_k]  # Top uncertain variables
    else:
        # Fallback: just use variables currently "on" plus some random ones
        import random
        on_vars = np.where(x_int == 1)[0]
        top_k = max(len(on_vars), int(m * polish_pool))
        candidates = random.sample(range(m), min(top_k, m))
    
    # Always include variables currently "on"
    on_vars = set(np.where(x_int == 1)[0])
    candidates = np.array(list(set(candidates) | on_vars))
    
    print(f"[polish] Refining {len(candidates)}/{m} variables ({100*len(candidates)/m:.1f}%)")
    
    # Build restricted MIP
    try:
        import gurobipy as gp
        from gurobipy import GRB
        
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.setParam('TimeLimit', polish_time)
            env.start()
            
            with gp.Model(env=env) as model:
                # Variables: only the candidates are free, others are fixed
                x = model.addVars(m, vtype=GRB.BINARY, name="x")
                
                # Fix variables NOT in candidates
                for j in range(m):
                    if j not in candidates:
                        x[j].lb = x[j].ub = int(x_int[j])
                
                # Set warm start from current solution
                for j in range(m):
                    x[j].Start = int(x_int[j])
                
                # Coverage constraints
                for i in range(n):
                    idx = np.where(A[i, :] == 1)[0]
                    if len(idx) > 0:
                        model.addConstr(gp.quicksum(x[j] for j in idx) >= 1.0)
                
                # Objective
                model.setObjective(gp.quicksum(c[j] * x[j] for j in range(m)), GRB.MINIMIZE)
                
                # Optimize
                model.optimize()
                
                if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
                    x_polished = np.array([x[j].X for j in range(m)])
                    x_polished = np.round(x_polished).astype(int)
                    cost_polished = float(c @ x_polished)
                    
                    elapsed = time.time() - t0
                    improvement = (c @ x_int) - cost_polished
                    if improvement > 0.01:
                        print(f"[polish] Improved: {c @ x_int:.2f} → {cost_polished:.2f} (-{improvement:.2f}, {elapsed:.3f}s)")
                    else:
                        print(f"[polish] No improvement ({elapsed:.3f}s)")
                    return x_polished, cost_polished, elapsed
                else:
                    print(f"[polish] Failed (status={model.Status})")
                    return x_int, float(c @ x_int), time.time() - t0
                    
    except Exception as e:
        print(f"[polish] Error: {e}")
        return x_int, float(c @ x_int), time.time() - t0

def solve_mip(A, c, timelimit_s=30, gurobi_time_limit=None, gurobi_gap_limit=None, track_gurobi_anytime=False):
    """
    Solve set cover MIP with optional anytime tracking.
    
    Args:
        A: Constraint matrix (n x m)
        c: Cost vector (m,)
        timelimit_s: Base time limit
        gurobi_time_limit: Specific Gurobi time limit (overrides timelimit_s)
        gurobi_gap_limit: MIP gap limit for Gurobi
        track_gurobi_anytime: If True, track incumbent progression
    
    Returns:
        dict with keys: obj, time, bound, gap, anytime_log (if tracking enabled)
    """
    sel = selected_solver()
    n, m = A.shape
    
    # Use specific time limit if provided
    actual_time_limit = gurobi_time_limit if gurobi_time_limit is not None else timelimit_s

    if sel == "gurobi":
        try:
            import os
            import gurobipy as gp
            from gurobipy import GRB
            
            anytime_log = [] if track_gurobi_anytime else None
            
            # Callback for anytime tracking
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    cur_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    cur_time = model.cbGet(GRB.Callback.RUNTIME)
                    anytime_log.append((cur_time, cur_obj, 'gurobi_incumbent'))
            
            t0 = time.time()
            env = gp.Env(empty=True)
    
            for name in ("WLSAccessID", "WLSSecret", "LicenseID", "WLSToken"):
                v = os.getenv(f"GRB_{name.upper()}")
                if v:
                    env.setParam(name, int(v) if name == "LicenseID" else v)
    
            for cand in ("gurobi.env", os.path.join(os.getcwd(), "gurobi.env")):
                if os.path.isfile(cand):
                    env.readParams(cand)
                    break
            
            if actual_time_limit is not None:
                env.setParam("TimeLimit", float(actual_time_limit))
            
            if gurobi_gap_limit is not None:
                env.setParam("MIPGap", float(gurobi_gap_limit))
            else:
                gap_env = os.getenv("GRB_MIPGAP") or os.getenv("MIP_GAP")
                if gap_env:
                    env.setParam("MIPGap", float(gap_env))
            
            thr_env = os.getenv("GRB_THREADS") or os.getenv("THREADS")
            if thr_env:
                env.setParam("Threads", int(thr_env))
    
            env.start()
            model = gp.Model(env=env)
            model.Params.OutputFlag = 0
    
            x = model.addMVar(shape=m, vtype=gp.GRB.BINARY, name="x")
    
            for i in range(n):
                idx = np.where(A[i, :] == 1)[0]
                if idx.size:
                    model.addConstr(x[idx].sum() >= 1.0)
    
            model.setObjective((c @ x), gp.GRB.MINIMIZE)
            
            # Optimize with or without callback
            if track_gurobi_anytime:
                model.optimize(callback)
            else:
                model.optimize()
            
            t1 = time.time()
    
            if model.Status not in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT):
                return {
                    'obj': np.nan, 
                    'time': t1 - t0, 
                    'bound': np.nan, 
                    'gap': np.nan,
                    'anytime_log': anytime_log
                }
    
            obj_val = float(model.ObjVal)
            try:
                bound = float(model.ObjBound)
                denom = max(abs(obj_val), 1e-9)
                gap_pct = 100.0 * max(0.0, (obj_val - bound) / denom)
            except Exception:
                bound, gap_pct = (np.nan, np.nan)
    
            return {
                'obj': obj_val, 
                'time': t1 - t0, 
                'bound': bound, 
                'gap': gap_pct,
                'anytime_log': anytime_log
            }
    
        except Exception as e:
            print(f"[synth] Gurobi failed: {e}; falling back to CBC.")
            sel = "cbc"

    # CBC fallback
    try:
        from ortools.linear_solver import pywraplp
    except Exception:
        return {'obj': np.nan, 'time': np.nan, 'bound': np.nan, 'gap': np.nan, 'anytime_log': None}

    solver = pywraplp.Solver.CreateSolver('CBC')
    if solver is None:
        return {'obj': np.nan, 'time': np.nan, 'bound': np.nan, 'gap': np.nan, 'anytime_log': None}

    x = [solver.BoolVar(f"x_{j}") for j in range(m)]
    for i in range(n):
        ct = solver.RowConstraint(1, solver.infinity(), f"cov_{i}")
        for j in np.where(A[i, :] == 1)[0]:
            ct.SetCoefficient(x[j], 1)
    
    obj = solver.Objective()
    for j in range(m):
        obj.SetCoefficient(x[j], float(c[j]))
    obj.SetMinimization()
    
    if actual_time_limit is not None:
        solver.SetTimeLimit(int(actual_time_limit * 1000))
    
    t0 = time.time()
    res = solver.Solve()
    t1 = time.time()
    
    if res not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return {'obj': np.nan, 'time': t1 - t0, 'bound': np.nan, 'gap': np.nan, 'anytime_log': None}
    
    obj_val = obj.Value()
    try:
        bound = solver.BestObjectiveBound()
        denom = max(abs(obj_val), 1e-9)
        gap_pct = 100.0 * max(0.0, (obj_val - bound) / denom)
    except Exception:
        bound, gap_pct = (np.nan, np.nan)
    
    return {'obj': obj_val, 'time': t1 - t0, 'bound': bound, 'gap': gap_pct, 'anytime_log': None}


def solve_entropy_setcover(A_matrix, c, tau=0.1, iters=50, tol=1e-3, tau_schedule=None,
                          polish_time=1.0, polish_pool=0.3):
    """
    Main entropy solver with optional polish refinement.
    """
    inst = _build_instance_from_matrix(A_matrix, c)
    
    # Parse tau schedule if provided
    taus = None
    if tau_schedule:
        try:
            taus = [float(t.strip()) for t in tau_schedule.split(",")]
        except:
            taus = None
    
    # Entropy relaxation with annealing
    x_frac, y, smooth_obj, t_relax, lin_cost, cov_min, cov_avg = entropy_relax_with_annealing(
        inst, tau=tau, iters=iters, tol=tol, tau_schedule=taus
    )
    
    # Dual-guided rounding
    x_int, cost, cov_min_int, feasible = round_cover_dual_guided(inst, x_frac, y)
    
    # Polish if requested
    total_time = t_relax
    if polish_time > 0 and feasible:
        x_int, cost, t_polish = polish_solution(
            x_int, A_matrix, c, 
            polish_time=polish_time, 
            polish_pool=polish_pool,
            x_frac=x_frac  # Use fractional solution to guide variable selection
        )
        total_time += t_polish
    
    return x_int, cost, feasible, total_time  # Note: total_time now includes polish

def run_family(scales, trials, out_path, tau=0.1, density=0.002, seed=42,
               with_mip=False, mip_timelimit=30, tau_schedule=None, iters=50, tol=1e-3):
    """Updated to use entropy framework with anytime tracking"""
    rows = []
    t0 = time.time()
    
    print(f"[synth] Using entropy framework: tau={tau}, iters={iters}, tol={tol}")
    if tau_schedule:
        print(f"[synth] Tau schedule: {tau_schedule}")
    
    for (n, m) in scales:
        for t in range(trials):
            s = (seed + t) % (2**32 - 1)
            A, cost = _gen_entropy_friendly_scp(n, m, seed=s)
            
            # Initialize anytime log for this trial
            anytime_log = []
            trial_start = time.time()
            
            # Use entropy solver instead of greedy
            t1 = time.time()
            x, hyb_obj, feas, t_relax = solve_entropy_setcover(
                A, cost, tau=tau, iters=iters, tol=tol, tau_schedule=tau_schedule
            )
            t2 = time.time()
            hyb_time = (t2 - t1)
            
            # Track hybrid algorithm completion
            elapsed = time.time() - trial_start
            if np.isfinite(hyb_obj):
                anytime_log.append((elapsed, hyb_obj, 'hybrid'))
            
            # MIP comparison with anytime tracking
            mip_obj = mip_time = best_bound = gap_pct = np.nan
            if with_mip:
                mip_result = solve_mip(
                    A, cost, 
                    timelimit_s=mip_timelimit, 
                    gurobi_time_limit=30.0, 
                    gurobi_gap_limit=None, 
                    track_gurobi_anytime=True
                )
                # ADD THIS DEBUG:
                print(f"[DEBUG SYNTH] mip_result type: {type(mip_result)}, has anytime_log: {'anytime_log' in mip_result if isinstance(mip_result, dict) else 'N/A'}")
                if isinstance(mip_result, dict) and 'anytime_log' in mip_result:
                    print(f"[DEBUG SYNTH] anytime_log length: {len(mip_result['anytime_log']) if mip_result['anytime_log'] else 0}")

                
                # Handle both old tuple return and new dict return
                if isinstance(mip_result, dict):
                    mip_obj = mip_result.get('obj', np.nan)
                    mip_time = mip_result.get('time', np.nan)
                    best_bound = mip_result.get('bound', np.nan)
                    gap_pct = mip_result.get('gap', np.nan)
                    
                    # Merge MIP anytime data if available
                    if 'anytime_log' in mip_result and mip_result['anytime_log']:
                        for t_mip, obj_mip, method_mip in mip_result['anytime_log']:
                            anytime_log.append((t_mip, obj_mip, f'gurobi_{method_mip}'))
                else:
                    # Old tuple format: (obj, time, bound, gap_pct)
                    mip_obj, mip_time, best_bound, gap_pct = mip_result
            
            # Verify feasibility
            if feas:
                coverage = A @ x
                actual_feas = int(np.all(coverage >= 1))
            else:
                actual_feas = 0
            
            # Build result row
            row = {
                "family": "SYN",
                "n": n, "m": m, "trial": t,
                "density": density, "tau": tau,
                "tau_schedule": tau_schedule,
                "iters": iters, "tol": tol,
                "hyb_int": hyb_obj,
                "hyb_time": hyb_time,
                "relax_time": t_relax,
                "feasible": actual_feas,
                "mip_obj": mip_obj,
                "mip_time": mip_time,
                "gap_pct": gap_pct,
            }
            
            # Add anytime performance data
            if anytime_log:
                for i, (t_point, obj, method) in enumerate(anytime_log):
                    row[f"anytime_t_{i}"] = float(t_point)
                    row[f"anytime_obj_{i}"] = float(obj) if np.isfinite(obj) else np.nan
                    row[f"anytime_method_{i}"] = str(method)
            
                # Also add a compact anytime summary
                anytime_summary = "|".join([f"{method}:{obj:.0f}@{t_point:.3f}s" 
                                           for t_point, obj, method in anytime_log 
                                           if np.isfinite(obj)])
                row["anytime_summary"] = anytime_summary
            
            rows.append(row)
            
            # Progress logging
            if with_mip and not np.isnan(mip_obj):
                gap = ((hyb_obj - mip_obj) / max(1e-9, abs(mip_obj))) * 100
                print(f"[{n}x{m}, t={t}] HYB={hyb_obj:.1f} vs MIP={mip_obj:.1f} (gap={gap:.1f}%)")
            else:
                print(f"[{n}x{m}, t={t}] HYB={hyb_obj:.1f} (feas={actual_feas})")
    
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[synth] wrote {out_path} rows={len(df)} wall={time.time()-t0:.2f}s")
    return df

"""
Warm Start Experiments: Use Hybrid solution to initialize Gurobi

Add this to your synth_setcover.py or rail_setcover.py
"""
import numpy as np
import time


def solve_mip_with_warmstart(A, c, x_initial, timelimit_s=30, gap_limit=None, 
                              threads=1, track_anytime=True):
    """
    Solve MIP with warm start from initial solution.
    
    Args:
        A: Constraint matrix (n x m)
        c: Cost vector (m,)
        x_initial: Initial binary solution from Hybrid (m,)
        timelimit_s: Time limit in seconds
        gap_limit: MIP gap tolerance
        threads: Number of threads
        track_anytime: Track incumbent progression
    
    Returns:
        dict with keys: obj, time, gap, status, anytime_log, improvement_time
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return {
            'obj': float('inf'),
            'time': 0.0,
            'gap': None,
            'status': 'NoGurobi',
            'anytime_log': [],
            'improvement_time': None
        }
    
    n, m = A.shape
    anytime_log = []
    improvement_time = None  # Time when Gurobi improves on warm start
    
    # Callback to track progression
    def callback(model, where):
        nonlocal improvement_time
        if where == GRB.Callback.MIPSOL:
            cur_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            cur_time = time.time() - start_time
            anytime_log.append((cur_time, cur_obj, 'gurobi_incumbent'))
            
            # Track when Gurobi first improves on warm start
            if improvement_time is None and cur_obj < initial_obj:
                improvement_time = cur_time
    
    start_time = time.time()
    
    # Build model
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        if threads is not None:
            env.setParam('Threads', int(threads))
        env.start()
        
        with gp.Model(env=env) as model:
            # Set parameters
            if timelimit_s is not None:
                model.setParam(GRB.Param.TimeLimit, float(timelimit_s))
            if gap_limit is not None:
                model.setParam(GRB.Param.MIPGap, float(gap_limit))
            
            # Variables
            x = model.addVars(m, vtype=GRB.BINARY, name="x")
            
            # Constraints
            for i in range(n):
                idx = np.where(A[i, :] == 1)[0]
                if len(idx) > 0:
                    model.addConstr(gp.quicksum(x[j] for j in idx) >= 1.0)
            
            # Objective
            model.setObjective(gp.quicksum(c[j] * x[j] for j in range(m)), GRB.MINIMIZE)
            
            # WARM START: Provide initial solution
            for j in range(m):
                x[j].Start = int(x_initial[j])
            
            # Verify initial solution quality
            initial_obj = float(np.dot(c, x_initial))
            anytime_log.append((0.0, initial_obj, 'warmstart_initial'))
            
            # Optimize with callback
            if track_anytime:
                model.optimize(callback)
            else:
                model.optimize()
            
            total_time = time.time() - start_time
            
            # Extract results
            if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
                final_obj = float(model.ObjVal)
                
                try:
                    bound = float(model.ObjBound)
                    gap = abs(final_obj - bound) / max(1e-9, abs(final_obj))
                except:
                    gap = None
                
                status = 'Optimal' if model.Status == GRB.OPTIMAL else 'TimeLimit'
                
                return {
                    'obj': final_obj,
                    'time': total_time,
                    'gap': gap,
                    'status': status,
                    'anytime_log': anytime_log,
                    'improvement_time': improvement_time,
                    'initial_obj': initial_obj,
                    'improvement': initial_obj - final_obj if improvement_time else 0.0
                }
            else:
                return {
                    'obj': float('inf'),
                    'time': total_time,
                    'gap': None,
                    'status': f'Status_{model.Status}',
                    'anytime_log': anytime_log,
                    'improvement_time': None,
                    'initial_obj': initial_obj,
                    'improvement': 0.0
                }


def run_warmstart_comparison(scales, trials, out_path, tau=0.1, seed=42,
                             mip_timelimit=30, tau_schedule=None, iters=50, tol=1e-3,
                             polish_time=1.0, polish_pool=0.3, rr_trials=5):
    """
    Compare three scenarios with full polish pipeline.
    """
    import pandas as pd
    import os
    
    rows = []
    
    print(f"[warmstart] Running warm-start comparison experiments")
    print(f"[warmstart] Scenarios: Hybrid-only, Gurobi-cold, Gurobi-warm")
    print(f"[warmstart] Polish: time={polish_time}s, pool={polish_pool}")
    
    for (n, m) in scales:
        for t in range(trials):
            s = (seed + t) % (2**32 - 1)
            A, cost = _gen_entropy_friendly_scp(n, m, seed=s)
            
            print(f"\n[{n}x{m}, trial={t}]")
            
            # Scenario 1: Hybrid only (with polish)
            t1 = time.time()
            x_hybrid, hyb_obj, feas, t_relax = solve_entropy_setcover(
                A, cost, 
                tau=tau, 
                iters=iters, 
                tol=tol, 
                tau_schedule=tau_schedule,
                polish_time=polish_time,    # NOW USED
                polish_pool=polish_pool     # NOW USED
            )
            hyb_time = time.time() - t1
            print(f"  Hybrid: obj={hyb_obj:.1f}, time={hyb_time:.3f}s")
            
            # Scenario 2: Gurobi cold start
            cold_result = solve_mip(
                A, cost,
                timelimit_s=mip_timelimit,
                gurobi_time_limit=mip_timelimit,
                track_gurobi_anytime=True
            )
            
            if isinstance(cold_result, dict):
                cold_obj = cold_result.get('obj', np.nan)
                cold_time = cold_result.get('time', np.nan)
            else:
                cold_obj, cold_time = cold_result[0], cold_result[1]
            
            print(f"  Gurobi-cold: obj={cold_obj:.1f}, time={cold_time:.3f}s")
            
            # Scenario 3: Gurobi warm start
            warm_result = solve_mip_with_warmstart(
                A, cost, x_hybrid,
                timelimit_s=mip_timelimit,
                track_anytime=True
            )
            
            warm_obj = warm_result['obj']
            warm_time = warm_result['time']
            improvement_time = warm_result.get('improvement_time', None)
            
            print(f"  Gurobi-warm: obj={warm_obj:.1f}, time={warm_time:.3f}s")
            if improvement_time is not None:
                print(f"    → Improved on Hybrid at {improvement_time:.3f}s")
            
            # Calculate metrics
            hybrid_vs_cold = ((hyb_obj - cold_obj) / max(1e-9, abs(cold_obj))) * 100
            warm_improvement = warm_result.get('improvement', 0.0)
            
            rows.append({
                'family': 'WARMSTART',
                'n': n,
                'm': m,
                'trial': t,
                'hybrid_obj': hyb_obj,
                'hybrid_time': hyb_time,
                'cold_obj': cold_obj,
                'cold_time': cold_time,
                'warm_obj': warm_obj,
                'warm_time': warm_time,
                'improvement_time': improvement_time if improvement_time else np.nan,
                'hybrid_gap_pct': hybrid_vs_cold,
                'warm_improvement': warm_improvement,
                'speedup_vs_cold': cold_time / warm_time if warm_time > 0 else np.nan,
            })
    
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n[warmstart] Wrote {out_path}, rows={len(df)}")
    
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scales", type=str, required=True,
                   help='Comma-separated sizes like "200x4000,400x8000"')
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--tau_schedule", type=str, default=None,
                   help='Comma-separated tau values like "0.5,0.2,0.1"')
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--density", type=float, default=0.002)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--with_mip", action="store_true")
    p.add_argument("--mip_timelimit", type=float, default=30.0)
    p.add_argument("--solver", choices=["cbc", "gurobi"], default="cbc",
                   help="MIP solver backend (default: cbc)")

    args = p.parse_args()
    
    global _SELECTED_SOLVER
    _os.environ["MIP_SOLVER"] = args.solver
    _SELECTED_SOLVER = args.solver.lower()

    scales = _parse_scales(args.scales)
    run_family(scales, args.trials, args.out,
               tau=args.tau, tau_schedule=args.tau_schedule,
               iters=args.iters, tol=args.tol,
               density=args.density, seed=args.seed,
               with_mip=args.with_mip, mip_timelimit=args.mip_timelimit)

if __name__ == "__main__":
    main()