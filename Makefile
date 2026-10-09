# Makefile for Burgers PINN Analysis

VENV_PYTHON := .venv/bin/python
PYTHON := $(shell if [ -f $(VENV_PYTHON) ]; then echo $(VENV_PYTHON); else echo python3; fi)
CONFIG ?= configs/default.yaml
export PYTHONPATH := .

.PHONY: help install train eval ablations ablations-fast all clean

help:
	@echo "Burgers PINN Production Suite"
	@echo "============================="
	@echo "  make train          Train model with Adam + L-BFGS staged optimization"
	@echo "  make eval           Evaluate model against benchmark and generate parity/fields plots"
	@echo "  make ablations      Run the 3 ablation sweeps and export markdown tables and plots"
	@echo "  make ablations-fast Run quick smoke-test ablation sweeps"
	@echo "  make all            Execute complete pipeline (train then eval)"
	@echo "  make install        Install dependencies from requirements.txt"
	@echo "  make clean          Remove checkpoints, figures, cached files, and results"

install:
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) scripts/train.py --config $(CONFIG)

eval:
	$(PYTHON) scripts/evaluate.py --config $(CONFIG)

ablations:
	$(PYTHON) scripts/run_ablations.py --config $(CONFIG)

ablations-fast:
	$(PYTHON) scripts/run_ablations.py --config $(CONFIG) --fast

all: train eval

clean:
	rm -f checkpoints/*.pt figures/*.png results/*.json results/*.md
	find . -type d -name "__pycache__" -exec rm -rf {} +
