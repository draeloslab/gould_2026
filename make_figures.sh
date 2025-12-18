#!/bin/bash

mkdir -p output

export PYTHONPATH="${PYTHONPATH}:$(pwd)/gould_2026"
COMMAND="python"

#$COMMAND figure_code/half_turn_plots/half_turn_log_pred_heatmaps.py --output output/hturn_log_pred_p_kf.png --pred_type kf
#$COMMAND figure_code/half_turn_plots/half_turn_log_pred_heatmaps.py --output output/hturn_log_pred_p_bw.png --pred_type bw
#$COMMAND figure_code/half_turn_plots/half_turn_log_pred_heatmaps.py --output output/hturn_log_pred_p_vjf.png --pred_type vjf


$COMMAND figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot time --output output/profile_by_step_prosvd.svg --transformer prosvd
$COMMAND figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot time --output output/profile_by_step_sjpca.svg --transformer sjpca
$COMMAND figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot time --output output/profile_by_step_mmica.svg --transformer mmica
$COMMAND figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot stability --output output/stability_by_step_prosvd.svg --transformer prosvd
$COMMAND figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot stability --output output/stability_by_step_sjpca.svg --transformer sjpca
$COMMAND figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot stability --output output/stability_by_step_mmica.svg --transformer mmica
$COMMAND figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot nearness_to_offline --output output/nearness_to_offline_by_step_prosvd.svg --transformer prosvd
$COMMAND figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot nearness_to_offline --output output/nearness_to_offline_by_step_sjpca.svg --transformer sjpca
$COMMAND figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot nearness_to_offline --output output/nearness_to_offline_by_step_mmica.svg --transformer mmica
$COMMAND figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot native_nearness_to_offline --output output/native_nearness_to_offline_by_step_prosvd.svg --transformer prosvd
$COMMAND figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot native_nearness_to_offline --output output/native_nearness_to_offline_by_step_sjpca.svg --transformer sjpca
$COMMAND figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot native_nearness_to_offline --output output/native_nearness_to_offline_by_step_mmica.svg --transformer mmica

#$COMMAND figure_code/simulation_plots/show_toy_dataset.py --output output/show_toy_dataset.svg
#
#$COMMAND figure_code/simulation_plots/learn_s_hat_toy.py --output output/learn_s_hat_toy_1step.svg --type-of-plot 1-step-prediction
#$COMMAND figure_code/simulation_plots/learn_s_hat_toy.py --output output/learn_s_hat_toy_manifold_error.svg --type-of-plot manifold-error
#
#$COMMAND figure_code/simulation_plots/learn_s_hat_ss.py --output output/ss_1step_kf.png --type-of-plot 1-step-prediction --type-of-predictor kf
#$COMMAND figure_code/simulation_plots/learn_s_hat_ss.py --output output/ss_1step_bw.png --type-of-plot 1-step-prediction --type-of-predictor bw
#$COMMAND figure_code/simulation_plots/learn_s_hat_ss.py --output output/ss_1step_vjf.png --type-of-plot 1-step-prediction --type-of-predictor vjf
#
#$COMMAND figure_code/simulation_plots/optimization_comparison.py --output output/optim_col_vs_rand.svg --type-of-plot optim_col_vs_rand
#$COMMAND figure_code/simulation_plots/optimization_comparison.py --output output/optim_open_vs_closed_toy.svg --type-of-plot optim_open_vs_closed_toy --type-of-dim-red prosvd # you could do more here
#$COMMAND figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_kf_prosvd_odoherty21.svg --type-of-plot optim_open_vs_closed --type-of-dim-red prosvd --type-of-autoreg kf --dataset odoherty21
#$COMMAND figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_kf_prosvd_zong22.svg --type-of-plot optim_open_vs_closed --type-of-dim-red prosvd --type-of-autoreg kf --dataset zong22
#$COMMAND figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_kf_sjpca_odoherty21.svg --type-of-plot optim_open_vs_closed --type-of-dim-red sjpca --type-of-autoreg kf --dataset odoherty21
#$COMMAND figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_kf_sjpca_zong22.svg --type-of-plot optim_open_vs_closed --type-of-dim-red sjpca --type-of-autoreg kf --dataset zong22
#$COMMAND figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_kf_mmica_odoherty21.svg --type-of-plot optim_open_vs_closed --type-of-dim-red mmica --type-of-autoreg kf --dataset odoherty21
#$COMMAND figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_kf_mmica_zong22.svg --type-of-plot optim_open_vs_closed --type-of-dim-red mmica --type-of-autoreg kf --dataset zong22
#$COMMAND figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_bw_prosvd_odoherty21.svg --type-of-plot optim_open_vs_closed --type-of-dim-red prosvd --type-of-autoreg bw --dataset odoherty21
#$COMMAND figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_bw_prosvd_zong22.svg --type-of-plot optim_open_vs_closed --type-of-dim-red prosvd --type-of-autoreg bw --dataset zong22
#$COMMAND figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_bw_sjpca_odoherty21.svg --type-of-plot optim_open_vs_closed --type-of-dim-red sjpca --type-of-autoreg bw --dataset odoherty21
#$COMMAND figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_bw_sjpca_zong22.svg --type-of-plot optim_open_vs_closed --type-of-dim-red sjpca --type-of-autoreg bw --dataset zong22
#$COMMAND figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_bw_mmica_odoherty21.svg --type-of-plot optim_open_vs_closed --type-of-dim-red mmica --type-of-autoreg bw --dataset odoherty21
#$COMMAND figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_bw_mmica_zong22.svg --type-of-plot optim_open_vs_closed --type-of-dim-red mmica --type-of-autoreg bw --dataset zong22
#$COMMAND figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_vjf_prosvd_odoherty21.svg --type-of-plot optim_open_vs_closed --type-of-dim-red prosvd --type-of-autoreg vjf --dataset odoherty21
#$COMMAND figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_vjf_prosvd_zong22.svg --type-of-plot optim_open_vs_closed --type-of-dim-red prosvd --type-of-autoreg vjf --dataset zong22
#$COMMAND figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_vjf_sjpca_odoherty21.svg --type-of-plot optim_open_vs_closed --type-of-dim-red sjpca --type-of-autoreg vjf --dataset odoherty21
#$COMMAND figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_vjf_sjpca_zong22.svg --type-of-plot optim_open_vs_closed --type-of-dim-red sjpca --type-of-autoreg vjf --dataset zong22
#$COMMAND figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_vjf_mmica_odoherty21.svg --type-of-plot optim_open_vs_closed --type-of-dim-red mmica --type-of-autoreg vjf --dataset odoherty21
#$COMMAND figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_vjf_mmica_zong22.svg --type-of-plot optim_open_vs_closed --type-of-dim-red mmica --type-of-autoreg vjf --dataset zong22
#$COMMAND figure_code/simulation_plots/optimization_comparison.py --output output/optim_col_vs_rand_with_high_d_rand.svg --type-of-plot optim_col_vs_rand_with_high_d_rand
#$COMMAND figure_code/simulation_plots/optimization_comparison.py --output output/optim_col_vs_rand_with_high_d_rand_closed.svg --type-of-plot optim_col_vs_rand_with_high_d_rand_closed


#papermill -p u_function 'curvy spins' -p output output/learn_s_hat_toy_1step_spin.svg figure_code/simulation_plots/learn_s_hat_toy_1step_spin.ipynb /tmp/output.ipynb
#papermill -p u_function 'curvy spins alld-resp' -p output output/learn_s_hat_toy_1step_spin_3d.svg figure_code/simulation_plots/learn_s_hat_toy_1step_spin.ipynb /tmp/output.ipynb

#$COMMAND figure_code/simulation_plots/zong_stim.py --output output/zong_stim.svg
#
#
#$COMMAND figure_code/paralell_compare_plots/paralell_compare.py --output output/paralell_compare.png
#$COMMAND figure_code/paralell_compare_plots/paralell_compare.py --output output/paralell_compare.svg
#
#$COMMAND figure_code/benchmarking/record_benchmark_run.py --output output/benchmark_prosvd_kf.csv --dimred_type prosvd --pred_type kf
#$COMMAND figure_code/benchmarking/record_benchmark_run.py --output output/benchmark_prosvd_bw.csv --dimred_type prosvd --pred_type bw
#$COMMAND figure_code/benchmarking/record_benchmark_run.py --output output/benchmark_prosvd_vjf.csv --dimred_type prosvd --pred_type vjf
#$COMMAND figure_code/benchmarking/record_benchmark_run.py --output output/benchmark_sjpca_kf.csv --dimred_type sjpca --pred_type kf
#$COMMAND figure_code/benchmarking/record_benchmark_run.py --output output/benchmark_sjpca_bw.csv --dimred_type sjpca --pred_type bw
#$COMMAND figure_code/benchmarking/record_benchmark_run.py --output output/benchmark_sjpca_vjf.csv --dimred_type sjpca --pred_type vjf
#$COMMAND figure_code/benchmarking/record_benchmark_run.py --output output/benchmark_mmica_kf.csv --dimred_type mmica --pred_type kf
#$COMMAND figure_code/benchmarking/record_benchmark_run.py --output output/benchmark_mmica_bw.csv --dimred_type mmica --pred_type bw
#$COMMAND figure_code/benchmarking/record_benchmark_run.py --output output/benchmark_mmica_vjf.csv --dimred_type mmica --pred_type vjf
#$COMMAND figure_code/benchmarking/benchmark_heatmap_table.py --output output/benchmark_heatmap_table.svg
#$COMMAND figure_code/benchmarking/benchmark_single_trace.py --input output/benchmark_sjpca_bw.csv --output output/benchmark_single_trace.svg