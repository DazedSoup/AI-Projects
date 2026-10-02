# cyberarena — run `make help`. Windows needs GNU make (e.g. `winget install ezwinports.make`).

PY_BOOT ?= py -3.12
ifeq ($(OS),Windows_NT)
  PY := .venv/Scripts/python.exe
else
  PY := .venv/bin/python
endif

EPISODES ?= 2000
SEED ?= 7
RUN ?= $(lastword $(sort $(wildcard runs/*)))

.PHONY: help venv install test lint data train-models arena enrich enrich-online dashboard clean

help:
	@echo "venv            create .venv with Python 3.12"
	@echo "install         install requirements + editable package"
	@echo "test / lint     pytest / ruff"
	@echo "data            download + preprocess datasets       (phase 2)"
	@echo "train-models    train the three classifiers          (phase 2)"
	@echo "arena           train red/blue agents, log episodes  (phase 3)  EPISODES=$(EPISODES) SEED=$(SEED)"
	@echo "enrich          SHAP + template rationale + MITRE    (phase 4)  RUN=$(RUN)  (offline, free)"
	@echo "enrich-online   same, Claude-written rationale       (phase 4)  needs ANTHROPIC_API_KEY, paid"
	@echo "dashboard       launch Streamlit                     (phase 5)"

venv:
	$(PY_BOOT) -m venv .venv

install: venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests

data:
	$(PY) -m cyberarena.ml.datasets --all

train-models:
	$(PY) -m cyberarena.ml.train --model all

arena:
	$(PY) -m cyberarena.arena.train --episodes $(EPISODES) --seed $(SEED)

enrich:
	$(PY) -m cyberarena.explain.enrich --run $(RUN)

enrich-online:
	$(PY) -m cyberarena.explain.enrich --run $(RUN) --online

dashboard:
	$(PY) -m streamlit run src/cyberarena/dashboard/app.py

clean:
	$(PY) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('runs', '.pytest_cache')]"
