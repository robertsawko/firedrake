#!/bin/sh

python elasticity.py 10 1 \
     -ksp_type fcg \
     -ksp_monitor_true_residual \
     -pc_type composite \
     -pc_composite_type additive \
     -pc_composite_pcs python,python \
     -sub_0_pc_python_type impl.PatchPC \
     -sub_0_sub_pc_type ilu \
     -scp_mat_type aij \
     -sub_1_pc_python_type impl.P1PC \
     -sub_1_lo_ksp_type preonly \
     -sub_1_lo_pc_type lu \
     -log_view



