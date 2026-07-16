ioailab_PACKAGE_VERSION := $(shell sed -n 's/^__version__ = "\(.*\)"/\1/p' src/ioailab/__init__.py)
ioailab_IMAGE_REPOSITORY ?= ioailab
ioailab_IMAGE_TAG ?= $(ioailab_PACKAGE_VERSION)
ioailab_IMAGE ?= $(ioailab_IMAGE_REPOSITORY):$(ioailab_IMAGE_TAG)
export ioailab_IMAGE

COMPOSE := docker compose -f docker/compose.yaml
SERVICE := dev
GUI_SERVICE := dev-gui
TY_CHECK := python -m ty check src examples tests --exclude .omx --exclude third_party --warn all --quiet

.PHONY: help image build shell shell-gui gui-up gui-ps gui-down gui-xhost-allow gui-xhost-deny python test lint format typecheck docs docs-watch docs-versions

help:
	@printf '%s\n' \
		'image      Print the version-tagged Docker image' \
		'build      Build the development image' \
		'shell      Open a shell in the dev container' \
		'shell-gui  Open a GUI shell; auto-mount GP001 when present' \
		'gui-up     Start the GUI dev container for IDE attach' \
		'gui-ps     Show the GUI dev container status and name' \
		'gui-down   Stop and remove the GUI dev container' \
		'gui-xhost-allow Allow local root X11 access for GUI Docker' \
		'gui-xhost-deny  Revoke local root X11 access for GUI Docker' \
		'python     Run Isaac Python in the dev container' \
		'test       Run pytest in the dev container' \
		'lint       Run ruff checks and advisory ty warnings in the dev container' \
		'format     Run ruff format in the dev container' \
		'typecheck  Run advisory ty check in the dev container' \
		'docs       Build the documentation site with mdBook' \
		'docs-watch Serve the documentation site locally with mdBook' \
		'docs-versions Build all doc versions with a version switcher'

image:
	@printf '%s\n' '$(ioailab_IMAGE)'

build:
	$(COMPOSE) build $(SERVICE)

shell:
	$(COMPOSE) run --rm $(SERVICE) bash


shell-gui:
	docker/shell_gui.sh bash

gui-up:
	$(COMPOSE) --profile gui up -d $(GUI_SERVICE)

gui-ps:
	$(COMPOSE) --profile gui ps -a $(GUI_SERVICE)

gui-down:
	$(COMPOSE) --profile gui stop $(GUI_SERVICE)
	$(COMPOSE) --profile gui rm -f $(GUI_SERVICE)

gui-xhost-allow:
	xhost +local:root

gui-xhost-deny:
	xhost -local:root

python:
	$(COMPOSE) run --rm $(SERVICE) python

test:
	$(COMPOSE) run --rm $(SERVICE) 'python -m pytest'

lint:
	$(COMPOSE) run --rm $(SERVICE) 'python -m ruff check .'
	$(COMPOSE) run --rm $(SERVICE) '$(TY_CHECK)'

format:
	$(COMPOSE) run --rm $(SERVICE) 'python -m ruff format .'

typecheck:
	$(COMPOSE) run --rm $(SERVICE) '$(TY_CHECK)'

docs:
	mdbook build

docs-watch:
	mdbook serve

docs-versions:
	scripts/build_versioned_docs.sh
