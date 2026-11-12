PYTHON ?= python3
VENV := .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: help setup run check
help:
	@printf 'make setup — tạo virtualenv và cài thư viện\nmake run   — mở giao diện ở 127.0.0.1:7860\nmake check — chạy kiểm thử offline\n'
setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -r requirements.txt
run:
	$(PY) app.py
check:
	$(PY) -m unittest discover -s tests -v
