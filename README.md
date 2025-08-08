# Bounded Controller Synthesis for CGBF

This repository is an add-on to the paper:

> Ding, J., Yuan, D. and Deka, S.A., 2025.  
> *Backstepping Reach-avoid Controller Synthesis for Multi-input Multi-output Systems with Mixed Relative Degrees*.  
> [arXiv:2505.03612](https://arxiv.org/abs/2505.03612)

It provides code and tools to **enforce control bounds** for the proposed reach-avoid controller via a Sum-of-Squares (SOS) programming approach.

**Further documentation:**  
See [`Docs/Bounded_Controller_Synthesis_for_CGBF_Controller.pdf`](./Docs/Bounded_Controller_Synthesis_for_CGBF_Controller.pdf)

---

## File Structure

- `Docs/`: Documentation of the proposed approach to enforce control bounds.
- `cbf_with_constraints.py`: Applies CBF to constrain dynamics within a defined safe $\psi$ region (Python).
- `constrained_CBF_solver.jl`: Julia script to solve the CBF SOS problem given system dynamics and safe region.
- `constrained_HOCBF_solver.jl`: Julia script to solve the High-Order CBF SOS problem.
- `sos_solver.jl`: Julia SOS solver providing:
    - `sos_solver`: Original SOS solver for the CGBF problem.
    - `sos_solver2`: Solver for control bounds with polynomial decision variable $\mu$.
    - `sos_solver3`: Solver for control bounds with scalar decision variable $\mu$.
- `combined_controller.jl`: CGBF controller with additional SOS constraints to enforce control bounds (depends on `sos_solver.jl`).
- `utils.py`: Python utility functions.
- `utils.jl`: Julia utility functions.

---

## References

1. Ding, J., Yuan, D. and Deka, S.A., 2025. *Backstepping Reach-avoid Controller Synthesis for Multi-input Multi-output Systems with Mixed Relative Degrees*. [arXiv:2505.03612](https://arxiv.org/abs/2505.03612)