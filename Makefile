.PHONY: reproduce simulation robustness figure1 test clean

SEEDS = 11 23 37 51 71

# Everything the paper reports, in dependency order.
reproduce: simulation robustness figure1
	@echo
	@echo "Figures written to:"
	@echo "  results/simulation/Figure_2_coverage.pdf"
	@echo "  results/simulation/Figure_3_mechanistic_validation.pdf"
	@echo "  results/simulation/Figure_4_few_shot_metrics.pdf"
	@echo "  results/robustness/Figure_S1_robustness_controls.pdf"
	@echo "  results/robustness/Figure_S2_sensitivity.pdf"
	@echo "  results/figure1/Figure_1_concept.pdf"

# Figures 2-4 and the metric summary.
simulation:
	uv run python run_simulation.py --workers 12

# Figures S1-S2. The sensitivity sweep uses three of the five seeds.
robustness:
	uv run python run_robustness.py --analysis core \
	  --seeds $(SEEDS) --output-dir results/robustness
	uv run python run_robustness.py --analysis sensitivity \
	  --seeds 11 23 37 --output-dir results/sensitivity
	uv run python make_robustness_figures.py \
	  --core results/robustness/robustness_by_training_run.csv \
	  --sensitivity results/sensitivity/robustness_by_training_run.csv \
	  --output-dir results/robustness

# Figure 1 is a schematic, not a simulation result. Requires pdflatex.
figure1:
	bash figure1/build_figure1.sh

test:
	uv run python test_perturbsim.py

clean:
	rm -rf results
