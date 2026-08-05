UV ?= uv
UCC_ARCHIVE ?= data/raw/5G-production-dataset.zip
UCC_MANIFEST ?= manifests/ucc_static_v1.json
AWGN_CONFIG ?= configs/awgn_calibration_v1.yaml
AWGN_PLAN ?= manifests/awgn_sweep_v1.json
AWGN_STATE ?= data/calibration_runs/awgn_campaign.json
AWGN_EXPORT ?= data/calibration_runs/awgn_campaign_export
export PYTHONPATH := $(CURDIR)/src

.PHONY: setup fetch-ucc curate-static static-report sweep-plan sweep-export test check

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

sweep-export:
	$(UV) run --locked rfsim-realism sweep-export --state $(AWGN_STATE) --config $(AWGN_CONFIG) --plan $(AWGN_PLAN) --output $(AWGN_EXPORT)

test:
	$(UV) run --locked pytest -q

check:
	$(UV) run --locked ruff check src tests
	$(UV) run --locked pytest -q
