PY ?= /home/xyane/software/.venv/bin/python
GEN ?= config/generation.yaml

.PHONY: install schemas generate validate enrich e2e simulate test clean

install:
	$(PY) -m pip install -e ".[dev]"

schemas:
	$(PY) scripts/export_schemas.py

generate:
	$(PY) scripts/generate_population.py

validate:
	$(PY) scripts/validate_population.py

enrich:
	$(PY) scripts/enrich_personas.py

e2e: schemas generate validate enrich
	@echo "E2E OK"

# Layer 3: behavioral scenario simulation (ROLE=mock is offline, no API key)
simulate:
	$(PY) scripts/simulate.py \
		--personas $(or $(PERSONAS),data/generated/personas/ae_uae_mock_v1_n1000_s42) \
		--scenarios config/scenarios/mock_scenarios.yaml \
		--role $(or $(ROLE),mock) \
		--out data/generated/simulations

test:
	$(PY) -m pytest -q

clean:
	rm -rf data/generated/* data/reports/* data/validated/*
