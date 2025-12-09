## 📝 Goal of the tutorial:
This tutorial aims to document the performance trade-offs of using PINNs and high-fidelity traditional numerical solvers from the SciML ecosystem.

## 🧪 Scope of Comparison:

The comparison will be using a multi parameter PDE like the Burgers' or Heat Equation, where a physical parameter is varied.

* Approach using PINNs : Train a single, continuous PINN model as a function of space, time, and the varied parameter.
* Traditional Numerical Approach: Use an optimized solver to generate solutions across the same parameter range.

**Status: Work In Progress (WIP)**
