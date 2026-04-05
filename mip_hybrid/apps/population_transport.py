# -*- coding: utf-8 -*-
"""
Applications benchmark:
  (A) Population synthesis / contingency tables (2D)
  (B) Transportation (min-cost flow, TU)
Implements: IPF/Sinkhorn relaxations + dependent rounding; cost-aware LP integerization.
"""

import time, csv, os
from dataclasses import dataclass
from typing import Optional, Tuple, List
#from ortools.graph import pywrapgraph
import numpy as np

# ---------- Optional backends ----------
_BACKEND = None
_HAS_GRAPH = False
try:
    from ortools.linear_solver import pywraplp
    _BACKEND = 'ortools'
    try:
        from ortools.graph import pywrapgraph
        _HAS_GRAPH = True
    except Exception:
        _HAS_GRAPH = False
except Exception:
    try:
        import pulp  # noqa
        _BACKEND = 'pulp'
    except Exception:
        _BACKEND = None

def backend_name() -> Optional[str]:
    return _BACKEND

# =========================================================
# A) POPULATION SYNTHESIS (2D CONTINGENCY TABLES)
# =========================================================

@dataclass
class Contingency2D:
    row_marg: np.ndarray  # (R,)
    col_marg: np.ndarray  # (C,)
    R: int
    C: int

def make_contingency2d(R:int, C:int, N:int, seed:int=1, skew:float=1.2) -> Contingency2D:
    """Build synthetic 2D marginals with mild skew; total N."""
    rng = np.random.default_rng(seed)
    r = rng.pareto(skew, size=R) + 0.5
    c = rng.pareto(skew, size=C) + 0.5
    r = r / r.sum(); c = c / c.sum()
    row = np.round(N * r).astype(int)
    diff = N - int(row.sum())
    if diff != 0:
        row[np.argmax(row)] += diff
    col = np.round(N * c).astype(int)
    diff = N - int(col.sum())
    if diff != 0:
        col[np.argmax(col)] += diff
    assert row.sum() == col.sum() == N
    return Contingency2D(row, col, R, C)

def ipf_2d(row_marg: np.ndarray, col_marg: np.ndarray, q: Optional[np.ndarray]=None,
           iters:int=1000, tol:float=1e-9) -> Tuple[np.ndarray, float]:
    """
    Classic 2D IPF / raking: KL projection onto row/col sums.
    Returns X (R,C) and wall-time.
    """
    t0 = time.time()
    R, C = len(row_marg), len(col_marg)
    if q is None:
        X = np.ones((R, C), dtype=float)
    else:
        X = q.copy().astype(float)
        X[X<=0] = 1e-12
    # scale to total
    X *= (row_marg.sum() / X.sum())
    for _ in range(iters):
        # rows
        rs = X.sum(axis=1)
        alpha = np.divide(row_marg, np.maximum(rs, 1e-16))
        X = (alpha[:,None]) * X
        # cols
        cs = X.sum(axis=0)
        beta = np.divide(col_marg, np.maximum(cs, 1e-16))
        X = X * (beta[None,:])
        # check
        if max(np.abs(X.sum(axis=1)-row_marg).max(),
               np.abs(X.sum(axis=0)-col_marg).max()) <= tol:
            break
    return X, (time.time()-t0)

# ---------- Population integerization: exact residual rounding (<1 per cell) ----------

def round_transport_min_cost_mcf(a: np.ndarray, b: np.ndarray, C: np.ndarray, topk: Optional[int]=None) -> Tuple[np.ndarray, float]:
    """
    Min-cost flow integerization (TU => integral). If OR-Tools graph is unavailable
    or a sparsified model proves infeasible, falls back to the dense LP version.
    Returns (Xint, round_time_seconds).
    """
    start = time.time()

    # Try OR-Tools graph first
    try:
        from ortools.graph import pywrapgraph
        m, n = len(a), len(b)

        # Optional sparsification: per-row keep top-k cheapest arcs.
        if topk is not None:
            row_keep = []
            for i in range(m):
                k = min(n, int(topk))
                js = np.argpartition(C[i], k-1)[:k]
                js = js[np.argsort(C[i, js])]
                row_keep.append(set(int(j) for j in js))
        else:
            row_keep = [None] * m

        smcf = pywrapgraph.SimpleMinCostFlow()
        S = m + n
        T = m + n + 1

        def add_arc(u, v, cap, cost):
            smcf.AddArcWithCapacityAndUnitCost(int(u), int(v), int(cap), int(cost))

        # Source -> suppliers
        for i in range(m):
            if a[i] > 0:
                add_arc(S, i, int(a[i]), 0)

        # Supplier -> demand arcs (capacities large; integer unit costs)
        scale = 1_000_000
        for i in range(m):
            J = range(n) if row_keep[i] is None else row_keep[i]
            for j in J:
                add_arc(i, m + int(j), int(b[j]), int(round(float(C[i, j]) * scale)))

        # Demands -> sink
        for j in range(n):
            if b[j] > 0:
                add_arc(m + j, T, int(b[j]), 0)

        # Supplies
        total = int(np.sum(a))
        node_count = m + n + 2
        supplies = [0] * node_count
        supplies[S] = total
        supplies[T] = -total
        for v, s in enumerate(supplies):
            smcf.SetNodeSupply(v, s)

        status = smcf.Solve()
        if status != smcf.OPTIMAL:
            # Fall back to LP if sparse MCF can’t route all flow
            Xlp, t_lp = round_transport_min_cost_lp(a, b, C)
            return Xlp, (time.time() - start)

        # Extract flow
        X = np.zeros((m, n), dtype=int)
        for e in range(smcf.NumArcs()):
            u = smcf.Tail(e); v = smcf.Head(e); f = smcf.Flow(e)
            if 0 <= u < m and m <= v < m + n and f > 0:
                j = v - m
                X[u, j] += int(f)

        # Sanity checks
        assert np.all(X.sum(axis=1) == a)
        assert np.all(X.sum(axis=0) == b)
        return X, (time.time() - start)

    except Exception:
        # No graph module (or any error) → dense LP
        Xlp, t_lp = round_transport_min_cost_lp(a, b, C)
        return Xlp, t_lp

def _dependent_round_2d_lp(X: np.ndarray,
                           row_marg: np.ndarray, col_marg: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    LP on residuals with 0–1 bounds (TU ⇒ integral). Per-cell deviation < 1.
    Objective favors larger fractional parts: min sum (1-frac)*z.
    """
    t0 = time.time()
    R, C = X.shape
    F = np.floor(X).astype(int)
    dr = (row_marg - F.sum(axis=1)).astype(int)
    dc = (col_marg - F.sum(axis=0)).astype(int)
    assert dr.sum() == dc.sum(), "Row/col deficits mismatch"

    frac = X - F
    if backend_name() == 'ortools':
        solver = pywraplp.Solver.CreateSolver('GLOP')  # LP is enough; TU + bounds -> integer
        z = [[solver.NumVar(0.0, 1.0, f"z_{i}_{j}") for j in range(C)] for i in range(R)]
        for i in range(R):
            ct = solver.RowConstraint(float(dr[i]), float(dr[i]), "")
            for j in range(C): ct.SetCoefficient(z[i][j], 1.0)
        for j in range(C):
            ct = solver.RowConstraint(float(dc[j]), float(dc[j]), "")
            for i in range(R): ct.SetCoefficient(z[i][j], 1.0)
        obj = solver.Objective()
        for i in range(R):
            for j in range(C):
                obj.SetCoefficient(z[i][j], float(1.0 - frac[i,j]))
        obj.SetMinimization()
        solver.Solve()
        Z = np.array([[z[i][j].solution_value() for j in range(C)] for i in range(R)])
    else:
        import pulp
        prob = pulp.LpProblem("residual_01", pulp.LpMinimize)
        z = [[pulp.LpVariable(f"z_{i}_{j}", lowBound=0, upBound=1, cat='Continuous') for j in range(C)] for i in range(R)]
        for i in range(R): prob += pulp.lpSum(z[i][j] for j in range(C)) == dr[i]
        for j in range(C): prob += pulp.lpSum(z[i][j] for i in range(R)) == dc[j]
        prob += pulp.lpSum((1.0 - frac[i,j]) * z[i][j] for i in range(R) for j in range(C))
        prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=10))
        Z = np.array([[pulp.value(z[i][j]) for j in range(C)] for i in range(R)])

    Z = np.rint(Z).astype(int)
    Xint = F + Z
    # checks
    assert np.all(Xint.sum(axis=1) == row_marg)
    assert np.all(Xint.sum(axis=0) == col_marg)
    assert float(np.abs(Xint - X).max()) < 1.0 + 1e-9
    return Xint, (time.time()-t0)

def _dependent_round_2d_lp_sparsified(
    X: np.ndarray,
    row_marg: np.ndarray,
    col_marg: np.ndarray,
    topk_extra: int = 8,
    eta: float = 0.02,
    eps_slack: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """
    Sparsified residual LP: build variables only on a candidate edge set
    cut down by (i) per-row top-K fractional cells and (ii) per-column top-K,
    with η-pruning. If the reduced LP is infeasible, FALLS BACK to the dense LP.

    Guarantees: exact marginals and per-cell deviation < 1 when eps_slack == 0.
    If eps_slack > 0, allows tiny marginal slack with an L1 penalty (kept off by default).
    """
    t0 = time.time()
    R, C = X.shape
    F = np.floor(X).astype(int)
    dr = (row_marg - F.sum(axis=1)).astype(int)
    dc = (col_marg - F.sum(axis=0)).astype(int)
    assert dr.sum() == dc.sum(), "Row/col deficits mismatch"

    frac = X - F

    # --- Build candidate mask E (R x C) ---
    E = np.zeros((R, C), dtype=bool)

    # (a) Per-ROW candidates
    for i in range(R):
        if dr[i] <= 0:
            continue
        # Start with η-qualified columns for this row
        J_eta = np.flatnonzero(frac[i] >= eta)
        # Ensure at least dr[i] candidates; add top-K if needed
        if J_eta.size < dr[i]:
            K = min(C, dr[i] + topk_extra)
            J_top = np.argpartition(-frac[i], K-1)[:K]
            J = np.unique(np.concatenate([J_eta, J_top]))
        else:
            K = min(J_eta.size, dr[i] + topk_extra)
            idx = np.argpartition(frac[i, J_eta], -(K))[-K:]
            J = J_eta[idx]
            # sort by descending fractional part (optional)
            J = J[np.argsort(-frac[i, J])]
        E[i, J] = True

    # (b) Per-COLUMN candidates
    for j in range(C):
        if dc[j] <= 0:
            continue
        I_eta = np.flatnonzero(frac[:, j] >= eta)
        if I_eta.size < dc[j]:
            Kc = min(R, dc[j] + topk_extra)
            I_top = np.argpartition(-frac[:, j], Kc-1)[:Kc]
            I = np.unique(np.concatenate([I_eta, I_top]))
        else:
            Kc = min(I_eta.size, dc[j] + topk_extra)
            idx = np.argpartition(frac[I_eta, j], -(Kc))[-Kc:]
            I = I_eta[idx]
            I = I[np.argsort(-frac[I, j])]
        E[I, j] = True

    # Quick feasibility sanity: rows/cols with positive deficit must have some edges
    if np.any((dr > 0) & (E.sum(axis=1) == 0)) or np.any((dc > 0) & (E.sum(axis=0) == 0)):
        # Too aggressive sparsification — fall back to dense LP
        return _dependent_round_2d_lp(X, row_marg, col_marg)

    # --- Build the reduced LP on edges in E ---
    use_ort = (backend_name() == 'ortools')
    try:
        if use_ort:
            solver = pywraplp.Solver.CreateSolver('GLOP')
            if solver is None:
                raise RuntimeError("Failed to create GLOP solver")

            # Variables: z_ij in [0,1] for (i,j) in E
            z = [[None]*C for _ in range(R)]
            for i in range(R):
                for j in np.flatnonzero(E[i]):
                    z[i][j] = solver.NumVar(0.0, 1.0, f"z_{i}_{j}")

            # Optional slack (kept off by default)
            if eps_slack > 0:
                s_row = [solver.NumVar(-eps_slack, eps_slack, f"s_row_{i}") for i in range(R)]
                s_col = [solver.NumVar(-eps_slack, eps_slack, f"s_col_{j}") for j in range(C)]
            else:
                s_row = s_col = None

            # Row constraints
            for i in range(R):
                if dr[i] == 0:
                    continue
                ct = solver.RowConstraint(float(dr[i]), float(dr[i]), "")
                has_any = False
                for j in np.flatnonzero(E[i]):
                    ct.SetCoefficient(z[i][j], 1.0); has_any = True
                if s_row is not None:
                    ct.SetCoefficient(s_row[i], 1.0)
                if not has_any:
                    # sparse model was too small — fallback
                    raise RuntimeError("Row has deficit but no candidate vars")

            # Column constraints
            for j in range(C):
                if dc[j] == 0:
                    continue
                ct = solver.RowConstraint(float(dc[j]), float(dc[j]), "")
                has_any = False
                for i in np.flatnonzero(E[:, j]):
                    ct.SetCoefficient(z[i][j], 1.0); has_any = True
                if s_col is not None:
                    ct.SetCoefficient(s_col[j], 1.0)
                if not has_any:
                    raise RuntimeError("Column has deficit but no candidate vars")

            # Objective: favor larger fractionals (min sum (1 - frac)*z), plus slack penalty if used
            obj = solver.Objective()
            for i in range(R):
                for j in np.flatnonzero(E[i]):
                    obj.SetCoefficient(z[i][j], float(1.0 - frac[i, j]))
            if s_row is not None and s_col is not None:
                # L1 penalty via two-sided variables is approximated here with small weight
                lam = 1e3  # tune if you enable slack
                for i in range(R): obj.SetCoefficient(s_row[i], lam)
                for j in range(C): obj.SetCoefficient(s_col[j], lam)
            obj.SetMinimization()

            status = solver.Solve()
            if status != pywraplp.Solver.OPTIMAL:
                # Fallback to dense LP
                return _dependent_round_2d_lp(X, row_marg, col_marg)

            Z = np.zeros_like(F, dtype=float)
            for i in range(R):
                for j in np.flatnonzero(E[i]):
                    Z[i, j] = z[i][j].solution_value()

        else:
            import pulp
            prob = pulp.LpProblem("residual_01_sparse", pulp.LpMinimize)

            # Variables on candidate edges
            z = {}
            for i in range(R):
                for j in np.flatnonzero(E[i]):
                    z[(i, j)] = pulp.LpVariable(f"z_{i}_{j}", lowBound=0, upBound=1, cat='Continuous')

            # Slack (optional)
            if eps_slack > 0:
                s_row = {i: pulp.LpVariable(f"s_row_{i}", lowBound=-eps_slack, upBound=eps_slack) for i in range(R)}
                s_col = {j: pulp.LpVariable(f"s_col_{j}", lowBound=-eps_slack, upBound=eps_slack) for j in range(C)}
            else:
                s_row = s_col = {}

            # Row constraints
            for i in range(R):
                if dr[i] == 0: 
                    continue
                vars_i = [z[(i, j)] for j in np.flatnonzero(E[i])]
                if not vars_i and eps_slack == 0:
                    return _dependent_round_2d_lp(X, row_marg, col_marg)
                prob += pulp.lpSum(vars_i) + (s_row.get(i, 0)) == float(dr[i])

            # Column constraints
            for j in range(C):
                if dc[j] == 0:
                    continue
                vars_j = [z[(i, j)] for i in np.flatnonzero(E[:, j])]
                if not vars_j and eps_slack == 0:
                    return _dependent_round_2d_lp(X, row_marg, col_marg)
                prob += pulp.lpSum(vars_j) + (s_col.get(j, 0)) == float(dc[j])

            # Objective
            obj = []
            for i in range(R):
                for j in np.flatnonzero(E[i]):
                    obj.append((1.0 - float(frac[i, j])) * z[(i, j)])
            if eps_slack > 0:
                lam = 1e3
                obj += [lam * s_row[i] for i in s_row]
                obj += [lam * s_col[j] for j in s_col]
            prob += pulp.lpSum(obj)

            prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=15))
            if pulp.LpStatus[prob.status] != "Optimal":
                return _dependent_round_2d_lp(X, row_marg, col_marg)

            Z = np.zeros_like(F, dtype=float)
            for (i, j), var in z.items():
                Z[i, j] = pulp.value(var)

        Z = np.rint(Z).astype(int)
        Xint = F + Z
        # checks
        assert np.all(Xint.sum(axis=1) == row_marg)
        assert np.all(Xint.sum(axis=0) == col_marg)
        assert float(np.abs(Xint - X).max()) < 1.0 + 1e-9
        return Xint, (time.time() - t0)

    except Exception:
        # Any construction/solve hiccup → dense LP
        return _dependent_round_2d_lp(X, row_marg, col_marg)


def _dependent_round_2d_mcf(X, row_marg, col_marg, cost_scale: int = 200_000, topk_extra: int = 8):
    """
    Residual 0–1 min-cost flow using OR-Tools SimpleMinCostFlow (faster when available).
    Per-cell deviation < 1 guaranteed by unit capacities.
    Falls back to the dense LP if the graph module is unavailable or infeasible.
    Returns (Xint, time_sec).
    """
    # Try to import graph; if unavailable, fall back to LP
    try:
        from ortools.graph import pywrapgraph
    except Exception:
        return _dependent_round_2d_lp(X, row_marg, col_marg)

    R, C = X.shape
    F = np.floor(X).astype(int)
    dr = (row_marg - F.sum(axis=1)).astype(int)
    dc = (col_marg - F.sum(axis=0)).astype(int)
    assert dr.sum() == dc.sum(), "Row/col deficits mismatch"

    frac = X - F
    start = time.time()

    # ---- Sparsify: take per-row top K by fractional part ----
    # K_i = dr_i + topk_extra, but not exceeding C
    row_edges = []
    for i in range(R):
        if dr[i] <= 0:
            row_edges.append(np.array([], dtype=int))
            continue
        K = min(C, int(dr[i]) + topk_extra)
        idx = np.argpartition(-frac[i], K-1)[:K]       # top-K (unsorted)
        idx = idx[np.argsort(-frac[i, idx])]           # sort by desc fraction
        row_edges.append(idx)

    smcf = pywrapgraph.SimpleMinCostFlow()
    S = R + C
    T = R + C + 1

    def add_arc(u, v, cap, cost):
        smcf.AddArcWithCapacityAndUnitCost(int(u), int(v), int(cap), int(cost))

    total = int(dr.sum())

    # Source -> rows
    for i in range(R):
        if dr[i] > 0:
            add_arc(S, i, int(dr[i]), 0)

    # Rows -> Cols (unit cap) over sparsified edges
    for i in range(R):
        idxs = row_edges[i]
        for j in idxs:
            w = max(0.0, 1.0 - float(frac[i, j]))
            add_arc(i, R + int(j), 1, int(round(w * cost_scale)))

    # Cols -> sink
    for j in range(C):
        if dc[j] > 0:
            add_arc(R + j, T, int(dc[j]), 0)

    # Supplies
    node_count = R + C + 2
    supplies = [0] * node_count
    supplies[S] = total
    supplies[T] = -total
    for v, b in enumerate(supplies):
        smcf.SetNodeSupply(v, b)

    status = smcf.Solve()
    # If infeasible (too aggressive sparsification) → fallback to dense LP
    if status != smcf.OPTIMAL or (smcf.OptimalCost() == 0 and total > 0):
        return _dependent_round_2d_lp(X, row_marg, col_marg)

    Z = np.zeros_like(F, dtype=int)
    for a in range(smcf.NumArcs()):
        u = smcf.Tail(a); v = smcf.Head(a); f = smcf.Flow(a)
        if 0 <= u < R and R <= v < R + C and f > 0:
            j = v - R
            Z[u, j] = 1

    Xint = F + Z
    # checks
    assert np.all(Xint.sum(axis=1) == row_marg)
    assert np.all(Xint.sum(axis=0) == col_marg)
    assert float(np.abs(Xint - X).max()) < 1.0 + 1e-9
    return Xint, time.time() - start

def _mask_from_topk_and_mass(C: np.ndarray, X: Optional[np.ndarray]=None,
                             topk: int = 30, eta: float = 0.0) -> np.ndarray:
    """Boolean mask E of candidate arcs: per-row top-k by C, plus X>=eta if provided."""
    m, n = C.shape
    E = np.zeros((m, n), dtype=bool)
    # per-row top-k by cost
    for i in range(m):
        k = min(n, int(topk))
        js = np.argpartition(C[i], k-1)[:k]
        E[i, js] = True
    # add high-mass arcs from entropic plan
    if X is not None and eta > 0.0:
        E |= (X >= eta)
    # ensure each positive-demand column has at least one arc
    for j in range(n):
        if not E[:, j].any():
            i = int(np.argmin(C[:, j]))
            E[i, j] = True
    return E

def round_transport_min_cost_lp_restricted(a: np.ndarray, b: np.ndarray,
                                           C: np.ndarray, E: np.ndarray) -> Tuple[np.ndarray, float, bool]:
    """
    Solve min-cost transport LP on restricted support E (bool mask).
    Returns (Xint, time_s, ok). If ok=False, caller should fall back to full LP.
    """
    t0 = time.time()
    m, n = len(a), len(b)
    if backend_name() == 'ortools':
        solver = pywraplp.Solver.CreateSolver('GLOP')
        if solver is None:
            return np.zeros((m,n), dtype=int), 0.0, False
        x = [[None]*n for _ in range(m)]
        # variables only on E
        for i in range(m):
            for j in range(n):
                if E[i, j]:
                    x[i][j] = solver.NumVar(0.0, solver.infinity(), f"x_{i}_{j}")
        # row sums
        for i in range(m):
            if a[i] == 0: continue
            ct = solver.RowConstraint(float(a[i]), float(a[i]), "")
            has = False
            for j in range(n):
                if x[i][j] is not None:
                    ct.SetCoefficient(x[i][j], 1.0); has = True
            if not has:  # no var in this row
                return np.zeros((m,n), dtype=int), time.time()-t0, False
        # col sums
        for j in range(n):
            if b[j] == 0: continue
            ct = solver.RowConstraint(float(b[j]), float(b[j]), "")
            has = False
            for i in range(m):
                if x[i][j] is not None:
                    ct.SetCoefficient(x[i][j], 1.0); has = True
            if not has:
                return np.zeros((m,n), dtype=int), time.time()-t0, False
        # objective
        obj = solver.Objective()
        for i in range(m):
            for j in range(n):
                if x[i][j] is not None:
                    obj.SetCoefficient(x[i][j], float(C[i, j]))
        obj.SetMinimization()
        stat = solver.Solve()
        if stat != pywraplp.Solver.OPTIMAL:
            return np.zeros((m,n), dtype=int), time.time()-t0, False
        X = np.zeros((m, n), dtype=float)
        for i in range(m):
            for j in range(n):
                if x[i][j] is not None:
                    X[i, j] = x[i][j].solution_value()
    else:
        import pulp
        prob = pulp.LpProblem("transport_costed_sparse", pulp.LpMinimize)
        x = {}
        for i in range(m):
            for j in range(n):
                if E[i, j]:
                    x[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", lowBound=0, cat='Continuous')
        # rows
        for i in range(m):
            vars_i = [x[(i, j)] for j in range(n) if (i, j) in x]
            if a[i] > 0 and not vars_i:
                return np.zeros((m,n), dtype=int), time.time()-t0, False
            if vars_i:
                prob += pulp.lpSum(vars_i) == float(a[i])
        # cols
        for j in range(n):
            vars_j = [x[(i, j)] for i in range(m) if (i, j) in x]
            if b[j] > 0 and not vars_j:
                return np.zeros((m,n), dtype=int), time.time()-t0, False
            if vars_j:
                prob += pulp.lpSum(vars_j) == float(b[j])
        prob += pulp.lpSum(C[i, j] * x[(i, j)] for (i, j) in x)
        prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=10))
        if pulp.LpStatus[prob.status] != "Optimal":
            return np.zeros((m,n), dtype=int), time.time()-t0, False
        X = np.zeros((m, n), dtype=float)
        for (i, j), var in x.items():
            X[i, j] = pulp.value(var)

    Xint = np.rint(X).astype(int)
    # feasibility checks
    if not (np.all(Xint.sum(axis=1) == a) and np.all(Xint.sum(axis=0) == b)):
        return np.zeros((m,n), dtype=int), time.time()-t0, False
    return Xint, (time.time()-t0), True

def round_transport_min_cost_approx(a: np.ndarray, b: np.ndarray, C: np.ndarray,
                                    Xtau: Optional[np.ndarray]=None,
                                    topk: int = 30, eta: float = 0.0) -> Tuple[np.ndarray, float, bool]:
    """
    Approximate transport rounder: restricted-support LP using top-k-by-cost per row,
    plus arcs with Xtau >= eta (if provided). Falls back to full LP on failure.
    Returns (Xint, time_s, used_restricted=True/False).
    """
    E = _mask_from_topk_and_mass(C, Xtau, topk=topk, eta=eta)
    Xint, t, ok = round_transport_min_cost_lp_restricted(a, b, C, E)
    if ok:
        return Xint, t, True
    # fallback
    Xint_full, t_full = round_transport_min_cost_lp(a, b, C)
    return Xint_full, t_full, False

def dependent_round_2d(
    X: np.ndarray,
    row_marg: np.ndarray,
    col_marg: np.ndarray,
    method: str = "mcf",
    sparsify: bool = False,
    topk_extra: int = 8,
    eta: float = 0.02,
    eps_slack: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """
    User-facing population rounder.
      - method="mcf": try SimpleMinCostFlow; if unavailable or infeasible, fall back to LP
                      (sparsified LP if sparsify=True, else dense LP)
      - method="lp" : use LP (sparsified if sparsify=True, else dense)
    """
    mth = method.lower()
    if mth == "mcf":
        # First try MCF; if it fails, use LP path
        try:
            from ortools.graph import pywrapgraph  # noqa: F401
            Xint, t = _dependent_round_2d_mcf(X, row_marg, col_marg)
            return Xint, t
        except Exception:
            # no graph module or failed MCF — proceed to LP selection
            pass

    # LP route
    if sparsify:
        return _dependent_round_2d_lp_sparsified(
            X, row_marg, col_marg,
            topk_extra=topk_extra, eta=eta, eps_slack=eps_slack
        )
    else:
        return _dependent_round_2d_lp(X, row_marg, col_marg)

def bench_population(
    R=80, C=80, N=50000, trials=3, seed=123,
    method: str = "mcf", sparsify: bool = True, topk_extra: int = 8, eta: float = 0.02
):
    """
    Population synthesis (2D): IPF relax + exact residual rounding.
    method: "mcf" tries SimpleMinCostFlow then falls back; "lp" forces LP path.
    sparsify: when using LP, build a reduced model (top-K + η) with safe fallback.
    """
    print("\n=== Population synthesis (2D) ===")
    rows = []

    # Determine label for logging
    rounder_label = None
    if method.lower() == "mcf":
        try:
            from ortools.graph import pywrapgraph  # noqa: F401
            rounder_label = "mcf"
        except Exception:
            rounder_label = "lp-sparse" if sparsify else "lp"
    else:
        rounder_label = "lp-sparse" if sparsify else "lp"

    for t in range(trials):
        inst = make_contingency2d(R, C, N, seed=seed + 100*t)
        # Relaxation: IPF
        X, t_relax = ipf_2d(inst.row_marg, inst.col_marg)
        # Integerization
        Xint, t_round = dependent_round_2d(
            X, inst.row_marg, inst.col_marg,
            method=method, sparsify=sparsify, topk_extra=topk_extra, eta=eta
        )

        # Diagnostics
        err_rows = int(np.abs(Xint.sum(axis=1) - inst.row_marg).max())
        err_cols = int(np.abs(Xint.sum(axis=0) - inst.col_marg).max())
        max_dev  = float(np.abs(Xint - X).max())

        print(f"[trial {t+1}] relax={t_relax:.03f}s, round={t_round:.03f}s, "
              f"marginal errors (row,col)=({err_rows},{err_cols}), max |Δ|={max_dev:.3f} "
              f"[rounder={rounder_label}]")

        rows.append({
            "family": "population2d",
            "R": R, "C": C, "N": N, "trial": t+1,
            "relax_time": t_relax, "round_time": t_round,
            "row_err": err_rows, "col_err": err_cols, "max_cell_dev": max_dev,
            "rounder": rounder_label
        })
    return rows

# =========================================================
# B) TRANSPORTATION (TU) — SINKHORN + COST-AWARE INTEGERIZATION
# =========================================================

@dataclass
class TransportInstance:
    supply: np.ndarray  # (m,)
    demand: np.ndarray  # (n,)
    C: np.ndarray       # costs (m,n)
    m: int
    n: int
    N: int

def make_transport(m:int, n:int, N:int, seed:int=1, cost_scale:float=1.0) -> TransportInstance:
    rng = np.random.default_rng(seed)
    # random integer supply/demand summing to N
    a = rng.dirichlet(np.ones(m)) * N
    b = rng.dirichlet(np.ones(n)) * N
    supply = np.round(a).astype(int); demand = np.round(b).astype(int)
    ds = N - int(supply.sum()); dd = N - int(demand.sum())
    if ds != 0: supply[np.argmax(supply)] += ds
    if dd != 0: demand[np.argmax(demand)] += dd
    C = cost_scale * rng.random((m,n))
    return TransportInstance(supply, demand, C, m, n, N)

def sinkhorn_balanced_uv(a: np.ndarray, b: np.ndarray, C: np.ndarray, tau: float=0.05,
                         iters:int=500, tol:float=1e-9) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Entropic OT with scalings. Returns (X, u, v, time)."""
    import time
    t0 = time.time()
    K = np.exp(-C / max(tau, 1e-12))
    u = np.ones_like(a, dtype=float)
    v = np.ones_like(b, dtype=float)
    for _ in range(iters):
        u = a / np.maximum(K @ v, 1e-18)
        v = b / np.maximum(K.T @ u, 1e-18)
        if np.max(np.abs((u*(K@v)) - a)) < tol and np.max(np.abs((v*(K.T@u)) - b)) < tol:
            break
    X = np.diag(u) @ K @ np.diag(v)
    return X, u, v, (time.time()-t0)


def sinkhorn_balanced(a: np.ndarray, b: np.ndarray, C: np.ndarray, tau: float=0.05,
                      iters:int=500, tol:float=1e-9) -> Tuple[np.ndarray,float]:
    """
    Entropic OT: min <C,X> + tau * KL(X||1) s.t. X1=a, X^T1=b, X>=0.
    Standard Sinkhorn.
    """
    t0 = time.time()
    K = np.exp(-C / max(tau,1e-12))
    u = np.ones_like(a, dtype=float)
    v = np.ones_like(b, dtype=float)
    for _ in range(iters):
        u = a / np.maximum(K @ v, 1e-18)
        v = b / np.maximum(K.T @ u, 1e-18)
        if np.max(np.abs((u*(K@v)) - a)) < tol and np.max(np.abs((v*(K.T@u)) - b)) < tol:
            break
    X = np.diag(u) @ K @ np.diag(v)
    return X, (time.time()-t0)

def round_transport_greedy_push(a,b,C,u,v):
    # a,b integer supplies/demands; u,v from Sinkhorn
    m,n = len(a), len(b)
    A = a.copy().astype(int); B = b.copy().astype(int)
    X = np.zeros((m,n), dtype=int)
    alpha = np.log(np.maximum(u,1e-18))  # any tau scaling just rescales r_ij ranking
    beta  = np.log(np.maximum(v,1e-18))
    for i in range(m):
        if A[i]==0: continue
        r = C[i] - alpha[i] - beta  # reduced costs (up to a positive scale)
        order = np.argsort(r)
        for j in order:
            if A[i]==0: break
            if B[j]==0: continue
            f = min(A[i], B[j])
            X[i,j] += f
            A[i]   -= f
            B[j]   -= f
    # Optional 1-shot repair if any leftover (should be rare with balanced totals)
    if A.sum()!=0 or B.sum()!=0:
        # build a tiny LP only on columns with B[j]>0 and rows with A[i]>0
        I = np.flatnonzero(A>0); J = np.flatnonzero(B>0)
        if I.size and J.size:
            mask = np.zeros_like(C, dtype=bool); mask[np.ix_(I,J)] = True
            Xrepair, _, ok = round_transport_min_cost_lp_restricted(a-X.sum(axis=1), 
                                                                    b-X.sum(axis=0), C, mask)
            if ok:
                X += Xrepair
    return X

def round_transport_min_cost_lp(a: np.ndarray, b: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray,float]:
    """
    Integerization by solving the min-cost transport LP with costs C.
    TU ⇒ LP solution is integral; returns Xint (integer) and wall time.
    """
    t0 = time.time()
    m, n = len(a), len(b)
    if backend_name() == 'ortools':
        solver = pywraplp.Solver.CreateSolver('GLOP')  # LP is sufficient (TU)
        x = [[solver.NumVar(0.0, solver.infinity(), f"x_{i}_{j}") for j in range(n)] for i in range(m)]
        # Row sums
        for i in range(m):
            ct = solver.RowConstraint(float(a[i]), float(a[i]), "")
            for j in range(n): ct.SetCoefficient(x[i][j], 1.0)
        # Column sums
        for j in range(n):
            ct = solver.RowConstraint(float(b[j]), float(b[j]), "")
            for i in range(m): ct.SetCoefficient(x[i][j], 1.0)
        # Objective: true costs
        obj = solver.Objective()
        for i in range(m):
            for j in range(n):
                obj.SetCoefficient(x[i][j], float(C[i, j]))
        obj.SetMinimization()
        solver.Solve()
        X = np.array([[x[i][j].solution_value() for j in range(n)] for i in range(m)])
    else:
        import pulp
        prob = pulp.LpProblem("transport_costed", pulp.LpMinimize)
        x = [[pulp.LpVariable(f"x_{i}_{j}", lowBound=0, cat='Continuous') for j in range(n)] for i in range(m)]
        for i in range(m): prob += pulp.lpSum(x[i][j] for j in range(n)) == a[i]
        for j in range(n): prob += pulp.lpSum(x[i][j] for i in range(m)) == b[j]
        prob += pulp.lpSum(C[i, j] * x[i][j] for i in range(m) for j in range(n))
        prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=30))
        X = np.array([[pulp.value(x[i][j]) for j in range(n)] for i in range(m)])
    Xint = np.rint(X).astype(int)   # Should already be integer by TU
    # Sanity: marginals exact
    assert np.all(Xint.sum(axis=1) == a)
    assert np.all(Xint.sum(axis=0) == b)
    return Xint, (time.time() - t0)

# Backward-compatible alias (if you previously called this name)
def round_transport_floor_residue_lp(a: np.ndarray, b: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray,float]:
    return round_transport_min_cost_lp(a, b, C)

def _reopt_from_greedy(a, b, C, Xgreedy, Xtau, topk, eta):
    """
    Re-optimize on a small support: union of
      - per-row top-k-by-cost,
      - Xtau >= eta arcs (if provided),
      - arcs used by the greedy solution.
    Falls back to full LP if the restricted model is infeasible.
    """
    E = _mask_from_topk_and_mass(C, Xtau, topk=topk, eta=eta)
    E |= (Xgreedy > 0)
    Xint, t_reopt, ok = round_transport_min_cost_lp_restricted(a, b, C, E)
    if ok:
        return Xint, t_reopt, True
    # fallback: full LP (still fast and integral by TU)
    Xfull, t_full = round_transport_min_cost_lp(a, b, C)
    return Xfull, t_reopt + t_full, False


def round_transport_greedy_push(a: np.ndarray, b: np.ndarray, C: np.ndarray,
                                u: np.ndarray, v: np.ndarray, tau: float,
                                topk: Optional[int]=None) -> Tuple[np.ndarray, float, bool]:
    """
    Greedy push by reduced cost r_ij = C_ij - tau*(log u_i + log v_j).
    Returns (Xint, time_s, ok). Falls back to full LP if infeasible.
    """
    import time
    t0 = time.time()
    m, n = len(a), len(b)
    A = a.astype(int).copy()
    B = b.astype(int).copy()
    X = np.zeros((m, n), dtype=int)
    alpha = tau * np.log(np.maximum(u, 1e-18))
    beta  = tau * np.log(np.maximum(v, 1e-18))

    for i in range(m):
        if A[i] == 0: continue
        r = C[i] - alpha[i] - beta  # smaller is better
        if topk is not None and topk < n:
            J = np.argpartition(r, topk-1)[:topk]
            J = J[np.argsort(r[J])]
        else:
            J = np.argsort(r)
        for j in J:
            if A[i] == 0: break
            if B[j] == 0: continue
            f = min(A[i], B[j])
            X[i, j] += f
            A[i]    -= f
            B[j]    -= f

    # If any residue remains, try a tiny restricted LP repair on leftover rows/cols
    if A.sum() != 0 or B.sum() != 0:
        I = np.flatnonzero(A > 0)
        J = np.flatnonzero(B > 0)
        if I.size and J.size:
            mask = np.zeros_like(C, dtype=bool)
            mask[np.ix_(I, J)] = True
            Xrep, t_rep, ok = round_transport_min_cost_lp_restricted(
                a - X.sum(axis=1), b - X.sum(axis=0), C, mask
            )
            if ok:
                X += Xrep
                A = a - X.sum(axis=1)
                B = b - X.sum(axis=0)

    # If still infeasible (rare), fall back to full LP to keep demo robust.
    if not (np.all(X.sum(axis=1) == a) and np.all(X.sum(axis=0) == b)):
        X, t_lp = round_transport_min_cost_lp(a, b, C)
        return X, (time.time() - t0) + t_lp, False

    return X, (time.time() - t0), True


def solve_transport_opt(a: np.ndarray, b: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray,float,float]:
    """
    Solve the exact min-cost transport (LP with TU -> integral).
    Returns (X*, obj, time).
    """
    t0 = time.time()
    m, n = len(a), len(b)
    if backend_name() == 'ortools':
        solver = pywraplp.Solver.CreateSolver('GLOP')  # LP sufficient
        x = [[solver.NumVar(0.0, solver.infinity(), f"x_{i}_{j}") for j in range(n)] for i in range(m)]
        for i in range(m):
            ct = solver.RowConstraint(float(a[i]), float(a[i]), "")
            for j in range(n):
                ct.SetCoefficient(x[i][j], 1.0)
        for j in range(n):
            ct = solver.RowConstraint(float(b[j]), float(b[j]), "")
            for i in range(m):
                ct.SetCoefficient(x[i][j], 1.0)
        obj = solver.Objective()
        for i in range(m):
            for j in range(n):
                obj.SetCoefficient(x[i][j], float(C[i,j]))
        obj.SetMinimization()
        solver.Solve()
        X = np.array([[x[i][j].solution_value() for j in range(n)] for i in range(m)])
        T = time.time() - t0
        return X, float((C*X).sum()), T
    else:
        import pulp
        prob = pulp.LpProblem("transport", pulp.LpMinimize)
        x = [[pulp.LpVariable(f"x_{i}_{j}", lowBound=0, cat='Continuous') for j in range(n)] for i in range(m)]
        for i in range(m):
            prob += pulp.lpSum(x[i][j] for j in range(n)) == a[i]
        for j in range(n):
            prob += pulp.lpSum(x[i][j] for i in range(m)) == b[j]
        prob += pulp.lpSum(C[i,j]*x[i][j] for i in range(m) for j in range(n))
        prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=30))
        X = np.array([[pulp.value(x[i][j]) for j in range(n)] for i in range(m)])
        T = time.time() - t0
        return X, float((C*X).sum()), T

def bench_transport(m=50, n=50, N=5000, trials=3, seed=123, tau=0.05,
                    mode: str = "exact", approx: Optional[bool] = None,
                    topk: int = 30, eta: float = 0.0):
    """
    Transport benchmark with three integerization modes:
      - mode="exact": full min-cost LP (TU -> integral)
      - mode="approx": restricted-support LP (top-k per row + Xtau>=eta), with safe fallback
      - mode="greedy": reduced-cost greedy push using Sinkhorn (u,v) + tiny LP repair;
                       safe fallback to approx, then exact.

    Back-compat: if 'approx' is provided (True/False) and 'mode' is left as default,
                 we map approx=True -> "approx", approx=False -> "exact".
    """
    # Back-compat mapping
    if approx is not None and mode == "exact":
        mode = "approx" if approx else "exact"
    mode = mode.lower()

    print("\n=== Transportation (TU) ===")
    rows = []
    for t in range(trials):
        inst = make_transport(m, n, N, seed=seed + 100*t)

        # Optimal (LP; TU ⇒ integral)
        Xopt, opt_cost, t_opt = solve_transport_opt(inst.supply, inst.demand, inst.C)

        # Entropic relaxation (always get X and (u,v) scalings once)
        Xtau, u, v, t_relax = sinkhorn_balanced_uv(inst.supply, inst.demand, inst.C, tau=tau)

        # Integerization by mode
        used = None
        if mode == "greedy":
            # fast reduced-cost push; small repair is inside the routine
            Xg, t_greedy, ok_g = round_transport_greedy_push(
                inst.supply, inst.demand, inst.C, u=u, v=v, tau=tau, topk=topk
            )
            # One-shot re-optimization on a small support (removes most of the gap)
            Xint, t_reopt, ok_re = _reopt_from_greedy(
                inst.supply, inst.demand, inst.C, Xg, Xtau, topk=topk, eta=eta
            )
            t_round = t_greedy + t_reopt
            used = "greedy+reopt" if ok_re else "greedy→exact"
        elif mode == "approx":
            Xint, t_round, ok = round_transport_min_cost_approx(
                inst.supply, inst.demand, inst.C, Xtau=Xtau, topk=topk, eta=eta
            )
            used = "approx" if ok else "exact"
            if not ok:
                Xint, t_round = round_transport_min_cost_lp(inst.supply, inst.demand, inst.C)
        else:  # "exact"
            Xint, t_round = round_transport_min_cost_lp(inst.supply, inst.demand, inst.C)
            used = "exact"

        hyb_cost = float((inst.C * Xint).sum())
        gap_pct = 100.0 * (hyb_cost - opt_cost) / max(1e-9, opt_cost)

        mode_str = f"{mode}" if mode == used else f"{mode}→{used}"
        print(f"[trial {t+1}] OPT={opt_cost:.3f} (t={t_opt:.03f}s) | "
              f"HYB int={hyb_cost:.3f}, gap={gap_pct:.2f}% "
              f"(relax={t_relax:.03f}s, round={t_round:.03f}s, mode={mode_str})")

        rows.append({
            "family": "transport", "m": m, "n": n, "N": N, "trial": t+1,
            "tau": tau, "opt_cost": opt_cost, "opt_time": t_opt,
            "hyb_cost": hyb_cost, "hyb_gap_pct": gap_pct,
            "relax_time": t_relax, "round_time": t_round,
            "mode": mode_str, "topk": topk, "eta": eta
        })
    return rows


# =========================================================
# Driver
# =========================================================
# In mip_hybrid/apps/population_transport.py  
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
    
    print(f"[pop] Writing {len(rows)} rows with {len(fieldnames)} columns to {out_path}")
    
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    
    print(f"[pop] Successfully wrote {out_path}")            
    
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    
    # Multiple problem sizes for better experimental coverage
    pop_sizes = [(80, 80, 50000), (120, 120, 120000), (160, 160, 200000)]
    transport_sizes = [(100, 100, 10000), (200, 200, 20000), (300, 300, 30000)]
    
    all_rows = []
    
    # Run population synthesis experiments
    for R, C, N in pop_sizes:
        print(f"Running population synthesis: {R}x{C}, N={N}")
        pop_rows = bench_population(R=R, C=C, N=N, trials=args.trials, seed=args.seed)
        all_rows.extend(pop_rows)
    
    # Run transportation experiments  
    for m, n, N in transport_sizes:
        print(f"Running transportation: {m}x{n}, N={N}")
        tr_rows = bench_transport(m=m, n=n, N=N, trials=args.trials, seed=args.seed)
        all_rows.extend(tr_rows)
    
    write_csv(all_rows, args.out)

if __name__ == "__main__":  # Fix the syntax error
    main()