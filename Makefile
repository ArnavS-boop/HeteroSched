BUILD_DIR := build
NPROC := $(shell nproc)

.PHONY: all clean build run setup train traces

all: build

build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. && make -j$(NPROC)

clean:
	@rm -rf $(BUILD_DIR)

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

train:
	python3 ml/dataset_generator.py
	python3 ml/train_model.py

traces:
	@mkdir -p benchmarks/traces
	python3 benchmarks/trace_generator.py

run: build traces
	./$(BUILD_DIR)/heterosched_bench --trace benchmarks/traces/mixed_trace.json --output results.csv

plot:
	python3 scripts/plot_results.py --input results.csv --output-dir artifacts/
