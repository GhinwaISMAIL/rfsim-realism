UV ?= uv
UCC_ARCHIVE ?= data/raw/5G-production-dataset.zip
UCC_MANIFEST ?= manifests/ucc_static_v1.json
AWGN_CONFIG ?= configs/awgn_calibration_v1.yaml
AWGN_PLAN ?= manifests/awgn_sweep_v1.json
STATIC_GRID_CONFIG ?= configs/ucc_static_grid_v1.yaml
STATIC_GRID_PLAN ?= manifests/ucc_static_grid_v1.json
STATIC_MAPPING_CONFIG ?= configs/static_mapping_v1.json
STATIC_MAPPING_DATASET ?= ../rfsim-realism-data/datasets/ucc_static_awgn_safe_v2
STATIC_MAPPING_CAMPAIGN ?= ../rfsim-realism-data/campaigns/ucc_static_awgn_safe_v2/campaign_state.json
STATIC_MAPPING_OUTPUT ?= data/model_runs/ucc_static_awgn_safe_v2_mapping_v1
export PYTHONPATH := $(CURDIR)/src

.PHONY: setup fetch-ucc curate-static static-report sweep-plan grid-plan static-map test check

setup:
	$(UV) sync --extra dev --locked

fetch-ucc:
	$(UV) run --locked rfsim-realism fetch-ucc --output $(UCC_ARCHIVE)

curate-static:
	$(UV) run --locked rfsim-realism curate-static --dataset $(UCC_ARCHIVE) --output $(UCC_MANIFEST)

static-report:
	$(UV) run --locked rfsim-realism static-report --manifest $(UCC_MANIFEST) --output reports/static_trace_catalog.html

sweep-plan:
	$(UV) run --locked rfsim-realism sweep-plan --config $(AWGN_CONFIG) --output $(AWGN_PLAN)

grid-plan:
	$(UV) run --locked rfsim-realism grid-plan --config $(STATIC_GRID_CONFIG) --output $(STATIC_GRID_PLAN)

static-map:
	$(UV) run --locked rfsim-realism static-map \
		--dataset-dir $(STATIC_MAPPING_DATASET) \
		--selection-manifest manifests/ucc_static_grid_v2_safe.json \
		--campaign-state $(STATIC_MAPPING_CAMPAIGN) \
		--ucc-manifest manifests/ucc_static_v1.json \
		--comparison-contract configs/comparison_contract_v1.json \
		--config $(STATIC_MAPPING_CONFIG) \
		--output $(STATIC_MAPPING_OUTPUT)

test:
	$(UV) run --locked pytest -q

check:
	$(UV) run --locked ruff check src tests
	$(UV) run --locked pytest -q
