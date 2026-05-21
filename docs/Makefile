# Minimal makefile for Sphinx documentation

SPHINXOPTS   ?=
SPHINXBUILD  ?= uv run sphinx-build

SOURCEDIR    = source
BUILDDIR     = build

AUTOGENDIR   = 

# ------------------------------------------------------------
# Default target
# ------------------------------------------------------------
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help clean Makefile

# ------------------------------------------------------------
# Clean
# ------------------------------------------------------------
clean:
	@rm -rf "$(SOURCEDIR)/api_reference/_autogen"
	@rm -rf "$(SOURCEDIR)/api_reference/base/_autogen"
	@rm -rf "$(SOURCEDIR)/api_reference/physics/_autogen"
	@rm -rf "$(SOURCEDIR)/api_reference/physics/base/_autogen"
	@rm -rf "$(SOURCEDIR)/api_reference/physics/compton_scattering/_autogen"
	@rm -rf "$(SOURCEDIR)/api_reference/physics/coltrims/_autogen"
	@echo "Delete _autogen output"
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# ------------------------------------------------------------
# Catch-all target
# ------------------------------------------------------------
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)