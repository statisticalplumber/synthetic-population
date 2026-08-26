PY ?= /home/xyane/software/.venv/bin/python
GEN ?= config/generation.yaml

.PHONY: install schemas generate validate enrich e2e test clean

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

test:
	$(PY) -m pytest -q

clean:
	rm -rf data/generated/* data/reports/* data/validated/*
