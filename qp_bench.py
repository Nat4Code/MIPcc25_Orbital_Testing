#!/usr/bin/env python3
import os, sys, time, json, glob, hashlib
import networkx as nx
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB

# ----------------- your functions (kept) -----------------

def sum_edge_weights(graph, u, C):
    return sum(graph[u][v].get('weight', 1) for v in C if graph.has_edge(u, v))

def color_refinement_quasilinear(G, model, quadratics):
    color_classes = defaultdict(set)
    tmp_color_classes = defaultdict(set)
    node_color = {}
    count = 0

    for constr in model.getConstrs():
        tmp_color_classes[('l', constr.rhs, constr.sense)].add(constr.ConstrName)

    for i, constr in enumerate(model.getQConstrs()):
        tmp_color_classes[('q', constr.getAttr(GRB.Attr.QCRHS), constr.getAttr(GRB.Attr.QCSense))].add(f"q_cons{i}")

    for var in model.getVars():
        tmp_color_classes[(var.VType, var.OBJ, var.UB, var.LB)].add(var.varName)

    for color, nodes in tmp_color_classes.items():
        color_classes[count] = tmp_color_classes[color]
        for node in nodes:
            node_color[node] = count
        count += 1

    color_classes[count] = set(quadratics)
    for quad in quadratics:
        node_color[quad] = count
    count += 1

    for current_color in list(color_classes.keys()):
        tmp_color_classes = defaultdict(set)
        for node in color_classes[current_color]:
            degree = sum(G[node][neighbor].get('weight', 1) for neighbor in G.neighbors(node))
            tmp_color_classes[degree].add(node)

        if len(tmp_color_classes) > 1:
            tmp_color_classes.popitem()
            for color, nodes in tmp_color_classes.items():
                color_classes[count] = nodes
                for node in nodes:
                    node_color[node] = count
                    color_classes[current_color].remove(node)
                count += 1

    color_test = set(range(count))

    while color_test:
        current_color = color_test.pop()
        nodes_in_class = color_classes[current_color]
        adj_colors = {node_color[neighbor] for node in nodes_in_class for neighbor in G.neighbors(node)
                      if len(color_classes[node_color[neighbor]]) > 1}

        for compare_color in adj_colors:
            nodes_compare = color_classes[compare_color]
            tmp_color_classes = defaultdict(set)

            for node in nodes_compare:
                degree = sum_edge_weights(G, node, nodes_in_class)
                tmp_color_classes[degree].add(node)

            if len(tmp_color_classes) > 1:
                tmp_color_classes.popitem()
                for color, nodes in tmp_color_classes.items():
                    color_classes[count] = nodes
                    for node in nodes:
                        node_color[node] = count
                        color_classes[compare_color].remove(node)
                    color_test.add(count)
                    count += 1

        if len(nodes_in_class) == 1:
            node = next(iter(nodes_in_class))
            if G.has_node(node):
                G.remove_node(node)

    return {color: list(nodes) for color, nodes in color_classes.items()}, color_classes, node_color

def stabilize(G, color_classes, node_color, node):
    old_color = node_color[node]
    if len(color_classes[old_color]) == 1:
        return {c: list(ns) for c, ns in color_classes.items()}, color_classes, node_color

    count = max(color_classes.keys()) + 1
    color_classes[old_color].remove(node)
    node_color[node] = count
    color_classes[count].add(node)

    test_colors = {count}
    count += 1

    while test_colors:
        current_color = test_colors.pop()
        nodes_in_class = color_classes[current_color]

        adj_colors = {node_color[neighbor] for node in nodes_in_class for neighbor in G.neighbors(node)
                      if len(color_classes[node_color[neighbor]]) > 1}

        for compare_color in adj_colors:
            nodes_compare = color_classes[compare_color]
            tmp_color_classes = defaultdict(set)

            for node2 in nodes_compare:
                degree = sum_edge_weights(G, node2, nodes_in_class)
                tmp_color_classes[degree].add(node2)

            if len(tmp_color_classes) > 1:
                for color, nodes in tmp_color_classes.items():
                    for node2 in nodes:
                        color_classes[compare_color].remove(node2)
                        node_color[node2] = count
                    color_classes[count] = nodes
                    test_colors.add(count)
                    count += 1

        if len(nodes_in_class) == 1:
            node_to_remove = next(iter(nodes_in_class))
            if G.has_node(node_to_remove):
                G.remove_node(node_to_remove)

    return {c: list(ns) for c, ns in color_classes.items()}, color_classes, node_color

def build_gurobi_graph(model):
    G = nx.Graph()

    for var in model.getVars():
        G.add_node(var.varName, type='variable')

    quadratics = set()
    for i in range(len(model.getQConstrs())):
        expr = model.getQCRow(model.getQConstrs()[i])
        for j in range(expr.size()):
            quad_name = "q" + expr.getVar1(j).varName + expr.getVar2(j).varName
            if quad_name not in quadratics:
                quadratics.add(quad_name)
                G.add_node(quad_name, type='quadratic')
                G.add_edge(expr.getVar1(j).varName, quad_name, weight=1.0)
                G.add_edge(expr.getVar2(j).varName, quad_name, weight=1.0)

    for constr in model.getConstrs():
        cname = constr.ConstrName
        G.add_node(cname, type='constraint')
        lin = model.getRow(constr)
        for j in range(lin.size()):
            G.add_edge(lin.getVar(j).varName, cname, weight=lin.getCoeff(j))

    for i in range(len(model.getQConstrs())):
        cname = "q_cons" + str(i)
        G.add_node(cname, type='quadratic')
        expr = model.getQCRow(model.getQConstrs()[i])
        lin = expr.getLinExpr()
        for j in range(lin.size()):
            G.add_edge(expr.getVar(j).varName, cname, weight=expr.getCoeff(j))
        for j in range(expr.size()):
            G.add_edge("q" + expr.getVar1(j).varName + expr.getVar2(j).varName, cname, weight=expr.getCoeff(j))

    return G, quadratics

def build_folded(relax, color_classes):
    # fold only variable partitions (by name convention in your code)
    for i in color_classes:
        if len(color_classes[i]) > 1:
            first = list(color_classes[i])[0]
            if first and (first[0] == 'b' or first[0] == 'x'):
                first_var = relax.getVarByName(first)
                for j in range(1, len(color_classes[i])):
                    second_var = relax.getVarByName(list(color_classes[i])[j])
                    relax.addConstr(first_var - second_var == 0)

def fix_vars(relax, tight, G):
    fixed0 = set()
    for v in relax.getVars():
        if v.x <= 1e-5:
            fix_var = tight.getVarByName(v.varName)
            fix_var.UB = 0
            if G.has_node(v.varName):
                G.remove_node(v.varName)
            fixed0.add(v.varName)
    return fixed0

def partition_fixings(color_classes, node_color, relax, G, fixed0, model, max_iters=100):
    best_color = 1
    fixings = set()
    iteration = 0

    while best_color >= 0 and iteration < max_iters:
        best_sum = 1.1
        best_color = -1
        best_node = None

        for color, nodes in color_classes.items():
            if len(nodes) > 1:
                first_node = next(iter(nodes))
                if first_node and first_node[0] == 'b':
                    v = relax.getVarByName(first_node)
                    if v is None:
                        continue
                    current_sum = v.x * len(nodes)
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_color = color
                        best_node = first_node

        if best_color < 0 or best_node is None:
            break

        fixings.add(best_node)
        old_size = len(color_classes[best_color])

        _, color_classes, node_color = stabilize(G, color_classes, node_color, best_node)
        new_size = len(color_classes[best_color])

        while best_sum * new_size / old_size > 5.0 and new_size > 1:
            best_sum -= 1
            best_node = next(iter(color_classes[best_color]))
            fixings.add(best_node)
            _, color_classes, node_color = stabilize(G, color_classes, node_color, best_node)
            new_size = len(color_classes[best_color])

        relax = model.relax()
        relax.setParam('OutputFlag', 0)

        for node in fixings:
            v = relax.getVarByName(node)
            if v is not None:
                v.LB = 1

        for node in fixed0:
            v = relax.getVarByName(node)
            if v is not None:
                v.UB = 0

        relax.update()
        build_folded(relax, color_classes)
        relax.optimize()

        iteration += 1

    return fixings

# ----------------- benchmarking additions -----------------

def orbit_summary_from_partition(color_classes, model):
    varnames = set(v.varName for v in model.getVars())
    orbit_sigs = []
    count = 0

    for _, nodes in color_classes.items():
        vars_in = sorted([n for n in nodes if n in varnames])
        if len(vars_in) > 1:
            orbit_sigs.append(",".join(vars_in))
            count += 1

    orbit_sigs.sort()
    allsig = "|".join(orbit_sigs)
    fp = hashlib.md5(allsig.encode("utf-8")).hexdigest()  # stable
    return count, fp

def status_name(st):
    mapping = {
        GRB.LOADED: "LOADED",
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INPROGRESS: "INPROGRESS",
    }
    return mapping.get(st, f"STATUS_{st}")

# ----------------- new: pass incumbent as a Gurobi MIP start -----------------

def extract_incumbent_by_name(m):
    """
    Return a dict varName -> X from model m, if it has any solution.
    Works for MIP/MIQP/MIQCP etc. If no solution, returns None.
    """
    try:
        if m.SolCount is None or m.SolCount <= 0:
            return None
    except gp.GurobiError:
        return None

    sol = {}
    for v in m.getVars():
        try:
            sol[v.varName] = float(v.X)
        except gp.GurobiError:
            # if X isn't available for some reason, skip
            pass
    return sol if sol else None

def apply_mip_start_by_name(m, sol_by_name):
    """
    Set Var.Start for vars whose names appear in sol_by_name.
    """
    if not sol_by_name:
        return
    for v in m.getVars():
        val = sol_by_name.get(v.varName, None)
        if val is not None:
            v.Start = val

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 qp_bench.py model.mps", file=sys.stderr)
        sys.exit(1)

    mps = sys.argv[1]
    t0 = time.time()

    model = gp.read(mps)
    model.setParam("OutputFlag", 0)
    #model.setParam("TimeLimit", 60.0)

    G, quadratics = build_gurobi_graph(model)
    partitions, color_classes, node_color = color_refinement_quasilinear(G, model, quadratics)

    orbit_count, orbit_fp = orbit_summary_from_partition(color_classes, model)

    relax = model.relax()
    relax.setParam("OutputFlag", 0)
    #relax.setParam("TimeLimit", 60.0)

    build_folded(relax, partitions)
    relax.optimize()

    tight_status = None
    tight_obj = None
    tight_runtime = None

    if relax.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        # --- Heuristic pipeline (same as before) ---
        tight_for_fix = model.copy()
        tight_for_fix.setParam("OutputFlag", 0)
        #tight_for_fix.setParam("TimeLimit", 60.0)

        fixed0 = fix_vars(relax, tight_for_fix, G)
        fixings = partition_fixings(color_classes, node_color, relax, G, fixed0, model)

        # Solve the "tightened" subproblem (this produces the heuristic incumbent)
        heuristic = model.copy()
        heuristic.setParam("OutputFlag", 0)
        #heuristic.setParam("TimeLimit", 60.0)

        for nm in fixings:
            v = heuristic.getVarByName(nm)
            if v is not None:
                v.LB = 1
        for nm in fixed0:
            v = heuristic.getVarByName(nm)
            if v is not None:
                v.UB = 0

        heuristic.update()
        heuristic.optimize()

        # --- NEW: pass the incumbent from `heuristic` to a full solve as a MIP start ---
        incumbent = extract_incumbent_by_name(heuristic)

        final = model.copy()
        final.setParam("OutputFlag", 0)
        #final.setParam("TimeLimit", 60.0)

        if incumbent is not None:
            apply_mip_start_by_name(final, incumbent)
            final.update()

        final.optimize()

        tight_status = final.status
        tight_runtime = final.Runtime
        if final.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) and final.SolCount > 0:
            tight_obj = float(final.ObjVal)

    wall = time.time() - t0

    out = {
        "tool": "qp_py",
        "mps": mps,
        "orbit_count": orbit_count,
        "orbit_fingerprint": orbit_fp,
        "tight_status_code": tight_status,
        "tight_status": status_name(tight_status) if tight_status is not None else None,
        "tight_obj": tight_obj,
        "tight_runtime_sec": tight_runtime,
        "wall_runtime_sec": wall,
    }
    print(json.dumps(out))

if __name__ == "__main__":
    main()
