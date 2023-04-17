# Makefile

all: serve
.PHONY: all

serve:
	python -m http.server
.PHONY: serve
