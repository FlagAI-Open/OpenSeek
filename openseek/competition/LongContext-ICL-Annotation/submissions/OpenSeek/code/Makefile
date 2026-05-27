PYTHON ?= python3
CONFIG ?= configs/base.yaml

.PHONY: prepare predict eval package test release-check release-check-dirty

prepare:
	./scripts/import_official_data.sh $(DATA_DIR)

predict:
	$(PYTHON) -m src.main predict --config $(CONFIG)

eval:
	$(PYTHON) -m src.main evaluate --config $(CONFIG)

package:
	$(PYTHON) -m src.main package --config $(CONFIG)

test:
	$(PYTHON) -m pytest -q

release-check:
	$(PYTHON) scripts/release_preflight.py

release-check-dirty:
	$(PYTHON) scripts/release_preflight.py --allow-dirty
