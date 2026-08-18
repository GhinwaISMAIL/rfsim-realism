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
RF_DISTRIBUTION_CONFIG ?= configs/rf_distribution_analysis_v1.yaml
RF_DISTRIBUTION_OUTPUT ?= data/model_runs/ucc_static_real_rf_catalog_v1
RF_DISTRIBUTION_MAPPING ?=
FAMILY_PRIMARY_DIR ?= ../rfsim-realism-data/model_runs/ucc_static_real_rf_catalog_v1_with_tdl_b_v2_support
FAMILY_CANDIDATE_DIR ?= data/model_runs/ucc_static_real_rf_catalog_v1_with_tdl_c_support
FAMILY_COMPARISON_OUTPUT ?= data/model_runs/ucc_static_tdl_b_tdl_c_family_comparison_v1
DISTRIBUTION_CALIBRATION_CONFIG ?= configs/distribution_calibration_tdl_b_v1.yaml
DISTRIBUTION_CALIBRATION_REAL ?= ../rfsim-realism-data/model_runs/ucc_static_real_rf_catalog_v1_with_tdl_b_v2_support/real_rf_observations.csv
DISTRIBUTION_CALIBRATION_EXECUTIONS ?= ../rfsim-realism-data/tdl_b_executions
DISTRIBUTION_CALIBRATION_SELECTION ?= ../rfsim-realism-data/campaigns/tdl_b_safe_v2/selection_manifest.json
DISTRIBUTION_CALIBRATION_CAMPAIGN ?= ../rfsim-realism-data/campaigns/tdl_b_safe_v2/campaign_state.json
DISTRIBUTION_CALIBRATION_OUTPUT ?= data/model_runs/ucc_static_tdl_b_distribution_calibration_v1
MMD_ABC_CONFIG ?= configs/mmd_abc_tdl_b_ploss_pilot_v1.yaml
MMD_ABC_PLAN ?= manifests/mmd_abc_tdl_b_ploss_pilot_v1.json
export PYTHONPATH := $(CURDIR)/src

.PHONY: setup fetch-ucc curate-static static-report sweep-plan grid-plan static-map rf-distribution distribution-calibrate mmd-abc-plan family-compare test check

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

rf-distribution:
	$(UV) run --locked rfsim-realism rf-distribution \
		--dataset $(UCC_ARCHIVE) \
		--manifest $(UCC_MANIFEST) $(if $(RF_DISTRIBUTION_MAPPING),--mapping-dir $(RF_DISTRIBUTION_MAPPING),) \
		--config $(RF_DISTRIBUTION_CONFIG) \
		--output $(RF_DISTRIBUTION_OUTPUT)

distribution-calibrate:
	$(UV) run --locked rfsim-realism distribution-calibrate \
		--real-observations $(DISTRIBUTION_CALIBRATION_REAL) \
		--executions-root $(DISTRIBUTION_CALIBRATION_EXECUTIONS) \
		--selection-manifest $(DISTRIBUTION_CALIBRATION_SELECTION) \
		--campaign-state $(DISTRIBUTION_CALIBRATION_CAMPAIGN) \
		--config $(DISTRIBUTION_CALIBRATION_CONFIG) \
		--output $(DISTRIBUTION_CALIBRATION_OUTPUT)

mmd-abc-plan:
	$(UV) run --locked rfsim-realism mmd-abc-plan \
		--config $(MMD_ABC_CONFIG) \
		--output $(MMD_ABC_PLAN)

family-compare:
	$(UV) run --locked rfsim-realism family-compare \
		--primary-dir $(FAMILY_PRIMARY_DIR) \
		--candidate-dir $(FAMILY_CANDIDATE_DIR) \
		--primary-label TDL_B \
		--candidate-label TDL_C \
		--output $(FAMILY_COMPARISON_OUTPUT)

test:
	$(UV) run --locked pytest -q

check:
	$(UV) run --locked ruff check src tests
	$(UV) run --locked pytest -q
