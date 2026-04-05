# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 09:28:11 2025

@author: rich
"""

from gurobipy import Model, GRB

# Create a model
m = Model()

# Add variables: x, y >= 0
x = m.addVar(lb=0, name="x")
y = m.addVar(lb=0, name="y")

# Set objective: minimize x + y
m.setObjective(x + y, GRB.MINIMIZE)

# Add constraint: x + 2y >= 10
m.addConstr(x + 2*y >= 10, "c1")

# Solve
m.optimize()

# Print results
if m.status == GRB.OPTIMAL:
    print(f"Objective value: {m.objVal}")
    print(f"x = {x.x}, y = {y.x}")
else:
    print("No optimal solution found.")
