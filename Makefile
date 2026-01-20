export PYTHONPATH:= $(PYTHONPATH):$(CURDIR)/gould_2026
export JAX_ENABLE_X64=1

.PHONY: all
TARGETS := $(shell grep -E '^output/' Makefile | cut -d: -f1 | sort -u)
all: $(TARGETS)
	coverage combine --keep
	coverage html
	@echo 'open output/reports/coverage-html/index.html 2>/dev/null'


COMMAND := coverage run --parallel-mode
PAPERMILL_COMMAND := coverage run --parallel-mode -m papermill
export JAX_PLATFORMS=cpu

output/hturn_log_pred_p_kf.png:
	$(COMMAND) figure_code/half_turn_plots/half_turn_log_pred_heatmaps.py --output output/hturn_log_pred_p_kf.png --pred_type kf
output/hturn_log_pred_p_bw.png:
	$(COMMAND) figure_code/half_turn_plots/half_turn_log_pred_heatmaps.py --output output/hturn_log_pred_p_bw.png --pred_type bw
output/hturn_log_pred_p_vjf.png:
	$(COMMAND) figure_code/half_turn_plots/half_turn_log_pred_heatmaps.py --output output/hturn_log_pred_p_vjf.png --pred_type vjf

output/profile_by_step_prosvd.svg:
	$(COMMAND) figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot time --output output/profile_by_step_prosvd.svg --transformer prosvd
output/profile_by_step_sjpca.svg:
	$(COMMAND) figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot time --output output/profile_by_step_sjpca.svg --transformer sjpca
output/profile_by_step_mmica.svg:
	$(COMMAND) figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot time --output output/profile_by_step_mmica.svg --transformer mmica
output/stability_by_step_prosvd.svg:
	$(COMMAND) figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot stability --output output/stability_by_step_prosvd.svg --transformer prosvd
output/stability_by_step_sjpca.svg:
	$(COMMAND) figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot stability --output output/stability_by_step_sjpca.svg --transformer sjpca
output/stability_by_step_mmica.svg:
	$(COMMAND) figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot stability --output output/stability_by_step_mmica.svg --transformer mmica
output/nearness_to_offline_by_step_prosvd.svg:
	$(COMMAND) figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot nearness_to_offline --output output/nearness_to_offline_by_step_prosvd.svg --transformer prosvd
output/nearness_to_offline_by_step_sjpca.svg:
	$(COMMAND) figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot nearness_to_offline --output output/nearness_to_offline_by_step_sjpca.svg --transformer sjpca
output/nearness_to_offline_by_step_mmica.svg:
	$(COMMAND) figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot nearness_to_offline --output output/nearness_to_offline_by_step_mmica.svg --transformer mmica
output/native_nearness_to_offline_by_step_prosvd.svg:
	$(COMMAND) figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot native_nearness_to_offline --output output/native_nearness_to_offline_by_step_prosvd.svg --transformer prosvd
output/native_nearness_to_offline_by_step_sjpca.svg:
	$(COMMAND) figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot native_nearness_to_offline --output output/native_nearness_to_offline_by_step_sjpca.svg --transformer sjpca
output/native_nearness_to_offline_by_step_mmica.svg:
	$(COMMAND) figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot native_nearness_to_offline --output output/native_nearness_to_offline_by_step_mmica.svg --transformer mmica

output/show_toy_dataset.svg:
	$(COMMAND) figure_code/simulation_plots/show_toy_dataset.py --output output/show_toy_dataset.svg

output/learn_s_hat_toy_1step.svg:
	$(COMMAND) figure_code/simulation_plots/learn_s_hat_toy.py --output output/learn_s_hat_toy_1step.svg --type-of-plot 1-step-prediction
output/learn_s_hat_toy_manifold_error.svg:
	$(COMMAND) figure_code/simulation_plots/learn_s_hat_toy.py --output output/learn_s_hat_toy_manifold_error.svg --type-of-plot manifold-error
#
output/ss_1step_kf.png:
	$(COMMAND) figure_code/simulation_plots/learn_s_hat_ss.py --output output/ss_1step_kf.png --type-of-plot 1-step-prediction --type-of-predictor kf
output/ss_1step_bw.png:
	$(COMMAND) figure_code/simulation_plots/learn_s_hat_ss.py --output output/ss_1step_bw.png --type-of-plot 1-step-prediction --type-of-predictor bw
output/ss_1step_vjf.png:
	$(COMMAND) figure_code/simulation_plots/learn_s_hat_ss.py --output output/ss_1step_vjf.png --type-of-plot 1-step-prediction --type-of-predictor vjf

output/optim_col_vs_rand.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output output/optim_col_vs_rand.svg --type-of-plot optim_col_vs_rand
output/optim_open_vs_closed_toy.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output output/optim_open_vs_closed_toy.svg --type-of-plot optim_open_vs_closed_toy --type-of-dim-red prosvd # you could do more here
output/open_vs_closed_by_dimred_kf_prosvd_odoherty21.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_kf_prosvd_odoherty21.svg --type-of-plot optim_open_vs_closed --type-of-dim-red prosvd --type-of-autoreg kf --dataset odoherty21
output/open_vs_closed_by_dimred_kf_prosvd_zong22.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_kf_prosvd_zong22.svg --type-of-plot optim_open_vs_closed --type-of-dim-red prosvd --type-of-autoreg kf --dataset zong22
output/open_vs_closed_by_dimred_kf_sjpca_odoherty21.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_kf_sjpca_odoherty21.svg --type-of-plot optim_open_vs_closed --type-of-dim-red sjpca --type-of-autoreg kf --dataset odoherty21
output/open_vs_closed_by_dimred_kf_sjpca_zong22.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_kf_sjpca_zong22.svg --type-of-plot optim_open_vs_closed --type-of-dim-red sjpca --type-of-autoreg kf --dataset zong22
output/open_vs_closed_by_dimred_kf_mmica_odoherty21.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_kf_mmica_odoherty21.svg --type-of-plot optim_open_vs_closed --type-of-dim-red mmica --type-of-autoreg kf --dataset odoherty21
output/open_vs_closed_by_dimred_kf_mmica_zong22.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_kf_mmica_zong22.svg --type-of-plot optim_open_vs_closed --type-of-dim-red mmica --type-of-autoreg kf --dataset zong22
output/open_vs_closed_by_dimred_bw_prosvd_odoherty21.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_bw_prosvd_odoherty21.svg --type-of-plot optim_open_vs_closed --type-of-dim-red prosvd --type-of-autoreg bw --dataset odoherty21
output/open_vs_closed_by_dimred_bw_prosvd_zong22.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_bw_prosvd_zong22.svg --type-of-plot optim_open_vs_closed --type-of-dim-red prosvd --type-of-autoreg bw --dataset zong22
output/open_vs_closed_by_dimred_bw_sjpca_odoherty21.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_bw_sjpca_odoherty21.svg --type-of-plot optim_open_vs_closed --type-of-dim-red sjpca --type-of-autoreg bw --dataset odoherty21
output/open_vs_closed_by_dimred_bw_sjpca_zong22.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_bw_sjpca_zong22.svg --type-of-plot optim_open_vs_closed --type-of-dim-red sjpca --type-of-autoreg bw --dataset zong22
output/open_vs_closed_by_dimred_bw_mmica_odoherty21.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_bw_mmica_odoherty21.svg --type-of-plot optim_open_vs_closed --type-of-dim-red mmica --type-of-autoreg bw --dataset odoherty21
output/open_vs_closed_by_dimred_bw_mmica_zong22.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_bw_mmica_zong22.svg --type-of-plot optim_open_vs_closed --type-of-dim-red mmica --type-of-autoreg bw --dataset zong22
output/open_vs_closed_by_dimred_vjf_prosvd_odoherty21.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_vjf_prosvd_odoherty21.svg --type-of-plot optim_open_vs_closed --type-of-dim-red prosvd --type-of-autoreg vjf --dataset odoherty21
output/open_vs_closed_by_dimred_vjf_prosvd_zong22.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_vjf_prosvd_zong22.svg --type-of-plot optim_open_vs_closed --type-of-dim-red prosvd --type-of-autoreg vjf --dataset zong22
output/open_vs_closed_by_dimred_vjf_sjpca_odoherty21.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_vjf_sjpca_odoherty21.svg --type-of-plot optim_open_vs_closed --type-of-dim-red sjpca --type-of-autoreg vjf --dataset odoherty21
output/open_vs_closed_by_dimred_vjf_sjpca_zong22.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_vjf_sjpca_zong22.svg --type-of-plot optim_open_vs_closed --type-of-dim-red sjpca --type-of-autoreg vjf --dataset zong22
output/open_vs_closed_by_dimred_vjf_mmica_odoherty21.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_vjf_mmica_odoherty21.svg --type-of-plot optim_open_vs_closed --type-of-dim-red mmica --type-of-autoreg vjf --dataset odoherty21
output/open_vs_closed_by_dimred_vjf_mmica_zong22.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output output/open_vs_closed_by_dimred_vjf_mmica_zong22.svg --type-of-plot optim_open_vs_closed --type-of-dim-red mmica --type-of-autoreg vjf --dataset zong22
output/optim_col_vs_rand_with_high_d_rand.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output output/optim_col_vs_rand_with_high_d_rand.svg --type-of-plot optim_col_vs_rand_with_high_d_rand
output/optim_col_vs_rand_with_high_d_rand_closed.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output output/optim_col_vs_rand_with_high_d_rand_closed.svg --type-of-plot optim_col_vs_rand_with_high_d_rand_closed
#
#
#
output/zong_stim.svg:
	$(COMMAND) figure_code/simulation_plots/zong_stim.py --output output/zong_stim.svg
#
#
output/paralell_compare.png:
	$(COMMAND) figure_code/paralell_compare_plots/paralell_compare.py --output output/paralell_compare.png
output/paralell_compare.svg:
	$(COMMAND) figure_code/paralell_compare_plots/paralell_compare.py --output output/paralell_compare.svg

output/parallel_compare_lpp_zoom.svg:
	$(PAPERMILL_COMMAND) -p output_file 'output/parallel_compare_lpp_zoom.svg' figure_code/paralell_compare_plots/switching_plot.ipynb /dev/null


#
output/benchmark_prosvd_kf.csv:
	$(COMMAND) figure_code/benchmarking/record_benchmark_run.py --output output/benchmark_prosvd_kf.csv --dimred_type prosvd --pred_type kf
output/benchmark_prosvd_bw.csv:
	$(COMMAND) figure_code/benchmarking/record_benchmark_run.py --output output/benchmark_prosvd_bw.csv --dimred_type prosvd --pred_type bw
output/benchmark_prosvd_vjf.csv:
	$(COMMAND) figure_code/benchmarking/record_benchmark_run.py --output output/benchmark_prosvd_vjf.csv --dimred_type prosvd --pred_type vjf
output/benchmark_sjpca_kf.csv:
	$(COMMAND) figure_code/benchmarking/record_benchmark_run.py --output output/benchmark_sjpca_kf.csv --dimred_type sjpca --pred_type kf
output/benchmark_sjpca_bw.csv:
	$(COMMAND) figure_code/benchmarking/record_benchmark_run.py --output output/benchmark_sjpca_bw.csv --dimred_type sjpca --pred_type bw
output/benchmark_sjpca_vjf.csv:
	$(COMMAND) figure_code/benchmarking/record_benchmark_run.py --output output/benchmark_sjpca_vjf.csv --dimred_type sjpca --pred_type vjf
output/benchmark_mmica_kf.csv:
	$(COMMAND) figure_code/benchmarking/record_benchmark_run.py --output output/benchmark_mmica_kf.csv --dimred_type mmica --pred_type kf
output/benchmark_mmica_bw.csv:
	$(COMMAND) figure_code/benchmarking/record_benchmark_run.py --output output/benchmark_mmica_bw.csv --dimred_type mmica --pred_type bw
output/benchmark_mmica_vjf.csv:
	$(COMMAND) figure_code/benchmarking/record_benchmark_run.py --output output/benchmark_mmica_vjf.csv --dimred_type mmica --pred_type vjf

output/benchmark_single_trace.svg: output/benchmark_sjpca_bw.csv
	$(COMMAND) figure_code/benchmarking/benchmark_single_trace.py --input output/benchmark_sjpca_bw.csv --output output/benchmark_single_trace.svg

#output/benchmark_heatmap_table.svg:
#	$(COMMAND) figure_code/benchmarking/benchmark_heatmap_table.py --output output/benchmark_heatmap_table.svg

output/learn_s_hat_toy_1step_spin.svg:
	export PYTHONPATH=PYTHONPATH:$(CURDIR)/figure_code/simulation_plots; \
	$(PAPERMILL_COMMAND) -p u_function 'curvy spins' -p output output/learn_s_hat_toy_1step_spin.svg figure_code/simulation_plots/learn_s_hat_toy_1step_spin.ipynb /tmp/output.ipynb

output/nonrotational_dynamics.svg:
	export PYTHONPATH=PYTHONPATH:$(CURDIR)/figure_code/simulation_plots; \
	$(PAPERMILL_COMMAND) -p output1 output/nonrotational_dynamics.svg -p output2 output/nonrotational_dynamics_sjpca_discovered_space.svg figure_code/simulation_plots/nonrotational_dynamics.ipynb /tmp/output.ipynb


output/learn_s_hat_toy_1step_spin_3d.svg:
	export PYTHONPATH=PYTHONPATH:$(CURDIR)/figure_code/simulation_plots; \
	$(PAPERMILL_COMMAND) -p u_function 'curvy spins alld-resp' -p output output/learn_s_hat_toy_1step_spin_3d.svg figure_code/simulation_plots/learn_s_hat_toy_1step_spin.ipynb /tmp/output.ipynb

output/starburst.svg:
	export PYTHONPATH=PYTHONPATH:$(CURDIR)/figure_code/simulation_plots; \
	$(PAPERMILL_COMMAND) -p output output/starburst.svg figure_code/simulation_plots/starburst.ipynb /tmp/output.ipynb



#papermill zong_stim_video for starburst