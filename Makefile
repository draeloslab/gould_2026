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
export JAX_ENABLE_X64=True

# fig:conceptual
# fig:spaces
output/paralell_compare.png:
	$(COMMAND) figure_code/paralell_compare_plots/paralell_compare.py --output $@
output/paralell_compare.svg:
	$(COMMAND) figure_code/paralell_compare_plots/paralell_compare.py --output $@


output/native_nearness_to_offline_by_step_prosvd.svg:
	$(COMMAND) figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot native_nearness_to_offline --output $@ --transformer prosvd
output/native_nearness_to_offline_by_step_sjpca.svg:
	$(COMMAND) figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot native_nearness_to_offline --output $@ --transformer sjpca
output/native_nearness_to_offline_by_step_mmica.svg:
	$(COMMAND) figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot native_nearness_to_offline --output $@ --transformer mmica


output/parallel_compare_lpp_zoom.svg:
	$(PAPERMILL_COMMAND) -p output_file 'output/parallel_compare_lpp_zoom.svg' figure_code/paralell_compare_plots/switching_plot.ipynb /dev/null

# fig:open
output/zong_stim.svg:
	$(COMMAND) figure_code/simulation_plots/zong_stim.py --output $@



output/starburst_constrained.svg:
	export PYTHONPATH=PYTHONPATH:$(CURDIR)/figure_code/simulation_plots; \
	$(PAPERMILL_COMMAND) -p output $@ -p condition constrained figure_code/simulation_plots/starburst.ipynb /tmp/output.ipynb

output/starburst_unconstrained.svg:
	export PYTHONPATH=PYTHONPATH:$(CURDIR)/figure_code/simulation_plots; \
	$(PAPERMILL_COMMAND) -p output $@ -p condition unconstrained figure_code/simulation_plots/starburst.ipynb /tmp/output.ipynb

output/starburst_positive_constrained.svg:
	export PYTHONPATH=PYTHONPATH:$(CURDIR)/figure_code/simulation_plots; \
	$(PAPERMILL_COMMAND) -p output $@ -p condition positive_constrained figure_code/simulation_plots/starburst.ipynb /tmp/output.ipynb

output/starburst_sparse_constrained.svg:
	export PYTHONPATH=PYTHONPATH:$(CURDIR)/figure_code/simulation_plots; \
	$(PAPERMILL_COMMAND) -p output $@ -p condition sparse_constrained figure_code/simulation_plots/starburst.ipynb /tmp/output.ipynb


# fig:toy
output/learn_s_hat_toy_1step_spin.svg:
	export PYTHONPATH=PYTHONPATH:$(CURDIR)/figure_code/simulation_plots; \
	$(PAPERMILL_COMMAND) -p u_function 'curvy spins' -p output $@ figure_code/simulation_plots/learn_s_hat_toy_1step_spin.ipynb /tmp/output.ipynb

output/learn_s_hat_toy_1step_spin_3d.svg:
	export PYTHONPATH=PYTHONPATH:$(CURDIR)/figure_code/simulation_plots; \
	$(PAPERMILL_COMMAND) -p u_function 'curvy spins alld-resp' -p output $@ figure_code/simulation_plots/learn_s_hat_toy_1step_spin.ipynb /tmp/output.ipynb

output/learn_s_hat_toy_manifold_error.svg:
	$(COMMAND) figure_code/simulation_plots/learn_s_hat_toy.py --output $@ --type-of-plot manifold-error


output/show_toy_dataset.svg:
	$(COMMAND) figure_code/simulation_plots/show_toy_dataset.py --output $@

# fig:mouse
output/daie21_1step.svg:
	$(PAPERMILL_COMMAND) -p output $@ figure_code/behavior/real_beh.ipynb /tmp/output.ipynb


# fig:fish
output/draelos25_1step.svg:
	$(PAPERMILL_COMMAND) \
	-p output1 output/draelos25_1step.svg \
	-p output2 output/draelos25_stim_locations.svg \
	-p output3 output/draelos25_low_and_low_d_response.svg \
	-p output4 output/draelos25_behavior.svg \
	-p output5 output/draelos25_behavior_stim_locations.svg \
	 figure_code/behavior/real_improv_dataset.ipynb /tmp/output.ipynb

# fig:closed
output/open_vs_closed_by_dimred_kf_prosvd_odoherty21.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output $@ --type-of-plot optim_open_vs_closed --type-of-dim-red prosvd --type-of-autoreg kf --dataset odoherty21

















output/hturn_log_pred_p_kf.png:
	$(COMMAND) figure_code/half_turn_plots/half_turn_log_pred_heatmaps.py --output $@ --pred_type kf
output/hturn_log_pred_p_bw.png:
	$(COMMAND) figure_code/half_turn_plots/half_turn_log_pred_heatmaps.py --output $@ --pred_type bw
output/hturn_log_pred_p_vjf.png:
	$(COMMAND) figure_code/half_turn_plots/half_turn_log_pred_heatmaps.py --output $@ --pred_type vjf

output/profile_by_step_prosvd.svg:
	$(COMMAND) figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot time --output $@ --transformer prosvd
output/profile_by_step_sjpca.svg:
	$(COMMAND) figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot time --output $@ --transformer sjpca
output/profile_by_step_mmica.svg:
	$(COMMAND) figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot time --output $@ --transformer mmica
output/stability_by_step_prosvd.svg:
	$(COMMAND) figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot stability --output $@ --transformer prosvd
output/stability_by_step_sjpca.svg:
	$(COMMAND) figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot stability --output $@ --transformer sjpca
output/stability_by_step_mmica.svg:
	$(COMMAND) figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot stability --output $@ --transformer mmica
output/nearness_to_offline_by_step_prosvd.svg:
	$(COMMAND) figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot nearness_to_offline --output $@ --transformer prosvd
output/nearness_to_offline_by_step_sjpca.svg:
	$(COMMAND) figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot nearness_to_offline --output $@ --transformer sjpca
output/nearness_to_offline_by_step_mmica.svg:
	$(COMMAND) figure_code/dim_red_plots/dimension_reduction_plots.py --type-of-plot nearness_to_offline --output $@ --transformer mmica


output/learn_s_hat_toy_1step.svg:
	$(COMMAND) figure_code/simulation_plots/learn_s_hat_toy.py --output $@ --type-of-plot 1-step-prediction

output/ss_1step_bw.png:
	$(COMMAND) figure_code/simulation_plots/learn_s_hat_ss.py --output $@ --type-of-plot 1-step-prediction --type-of-predictor bw
output/ss_1step_vjf.png:
	$(COMMAND) figure_code/simulation_plots/learn_s_hat_ss.py --output $@ --type-of-plot 1-step-prediction --type-of-predictor vjf

output/optim_col_vs_rand.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output $@ --type-of-plot optim_col_vs_rand
output/optim_open_vs_closed_toy.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output $@ --type-of-plot optim_open_vs_closed_toy --type-of-dim-red prosvd
output/open_vs_closed_by_dimred_kf_prosvd_zong22.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output $@ --type-of-plot optim_open_vs_closed --type-of-dim-red prosvd --type-of-autoreg kf --dataset zong22
output/open_vs_closed_by_dimred_kf_sjpca_odoherty21.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output $@ --type-of-plot optim_open_vs_closed --type-of-dim-red sjpca --type-of-autoreg kf --dataset odoherty21
output/open_vs_closed_by_dimred_kf_sjpca_zong22.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output $@ --type-of-plot optim_open_vs_closed --type-of-dim-red sjpca --type-of-autoreg kf --dataset zong22
output/open_vs_closed_by_dimred_kf_mmica_odoherty21.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output $@ --type-of-plot optim_open_vs_closed --type-of-dim-red mmica --type-of-autoreg kf --dataset odoherty21
output/open_vs_closed_by_dimred_kf_mmica_zong22.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output $@ --type-of-plot optim_open_vs_closed --type-of-dim-red mmica --type-of-autoreg kf --dataset zong22
output/open_vs_closed_by_dimred_bw_prosvd_odoherty21.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output $@ --type-of-plot optim_open_vs_closed --type-of-dim-red prosvd --type-of-autoreg bw --dataset odoherty21
output/open_vs_closed_by_dimred_bw_prosvd_zong22.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output $@ --type-of-plot optim_open_vs_closed --type-of-dim-red prosvd --type-of-autoreg bw --dataset zong22
output/open_vs_closed_by_dimred_bw_sjpca_odoherty21.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output $@ --type-of-plot optim_open_vs_closed --type-of-dim-red sjpca --type-of-autoreg bw --dataset odoherty21
output/open_vs_closed_by_dimred_bw_sjpca_zong22.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output $@ --type-of-plot optim_open_vs_closed --type-of-dim-red sjpca --type-of-autoreg bw --dataset zong22
output/open_vs_closed_by_dimred_bw_mmica_odoherty21.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output $@ --type-of-plot optim_open_vs_closed --type-of-dim-red mmica --type-of-autoreg bw --dataset odoherty21
output/open_vs_closed_by_dimred_bw_mmica_zong22.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output $@ --type-of-plot optim_open_vs_closed --type-of-dim-red mmica --type-of-autoreg bw --dataset zong22
output/open_vs_closed_by_dimred_vjf_prosvd_odoherty21.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output $@ --type-of-plot optim_open_vs_closed --type-of-dim-red prosvd --type-of-autoreg vjf --dataset odoherty21
output/open_vs_closed_by_dimred_vjf_prosvd_zong22.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output $@ --type-of-plot optim_open_vs_closed --type-of-dim-red prosvd --type-of-autoreg vjf --dataset zong22
output/open_vs_closed_by_dimred_vjf_sjpca_odoherty21.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output $@ --type-of-plot optim_open_vs_closed --type-of-dim-red sjpca --type-of-autoreg vjf --dataset odoherty21
output/open_vs_closed_by_dimred_vjf_sjpca_zong22.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output $@ --type-of-plot optim_open_vs_closed --type-of-dim-red sjpca --type-of-autoreg vjf --dataset zong22
output/open_vs_closed_by_dimred_vjf_mmica_odoherty21.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output $@ --type-of-plot optim_open_vs_closed --type-of-dim-red mmica --type-of-autoreg vjf --dataset odoherty21
output/open_vs_closed_by_dimred_vjf_mmica_zong22.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output $@ --type-of-plot optim_open_vs_closed --type-of-dim-red mmica --type-of-autoreg vjf --dataset zong22

output/compare_opt_by_target.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output $@ --type-of-plot compare_opt_by_target
output/compare_opt_by_target_closed.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output $@ --type-of-plot compare_opt_by_target --closed-loop


output/compare_opt_by_target_unconstrained.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output $@ --type-of-plot compare_opt_by_target --optimization-method jaxopt_unconstrained
output/compare_opt_by_target_closed_unconstrained.svg:
	$(COMMAND) figure_code/simulation_plots/optimization_comparison.py --output $@ --type-of-plot compare_opt_by_target --optimization-method jaxopt_unconstrained --closed-loop





output/benchmark_prosvd_kf.csv:
	$(COMMAND) figure_code/benchmarking/record_benchmark_run.py --output $@ --dimred_type prosvd --pred_type kf
output/benchmark_prosvd_bw.csv:
	$(COMMAND) figure_code/benchmarking/record_benchmark_run.py --output $@ --dimred_type prosvd --pred_type bw
output/benchmark_prosvd_vjf.csv:
	$(COMMAND) figure_code/benchmarking/record_benchmark_run.py --output $@ --dimred_type prosvd --pred_type vjf
output/benchmark_sjpca_kf.csv:
	$(COMMAND) figure_code/benchmarking/record_benchmark_run.py --output $@ --dimred_type sjpca --pred_type kf
output/benchmark_sjpca_bw.csv:
	$(COMMAND) figure_code/benchmarking/record_benchmark_run.py --output $@ --dimred_type sjpca --pred_type bw
output/benchmark_sjpca_vjf.csv:
	$(COMMAND) figure_code/benchmarking/record_benchmark_run.py --output $@ --dimred_type sjpca --pred_type vjf
output/benchmark_mmica_kf.csv:
	$(COMMAND) figure_code/benchmarking/record_benchmark_run.py --output $@ --dimred_type mmica --pred_type kf
output/benchmark_mmica_bw.csv:
	$(COMMAND) figure_code/benchmarking/record_benchmark_run.py --output $@ --dimred_type mmica --pred_type bw
output/benchmark_mmica_vjf.csv:
	$(COMMAND) figure_code/benchmarking/record_benchmark_run.py --output $@ --dimred_type mmica --pred_type vjf

output/benchmark_single_trace.svg: output/benchmark_sjpca_bw.csv
	$(COMMAND) figure_code/benchmarking/benchmark_single_trace.py --input output/benchmark_sjpca_bw.csv --output $@

output/benchmark_heatmap_table.svg:
	$(COMMAND) figure_code/benchmarking/benchmark_heatmap_table.py --output $@


output/nonrotational_dynamics.svg:
	export PYTHONPATH=PYTHONPATH:$(CURDIR)/figure_code/simulation_plots; \
	$(PAPERMILL_COMMAND) -p output1 $@ -p output2 output/nonrotational_dynamics_sjpca_discovered_space.svg figure_code/simulation_plots/nonrotational_dynamics.ipynb /tmp/output.ipynb

