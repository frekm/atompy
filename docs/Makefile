# Minimal makefile for Sphinx documentation

SPHINXOPTS   ?=
SPHINXBUILD  ?= sphinx-build

SOURCEDIR    = source
BUILDDIR     = build

AUTOGENDIR   = $(SOURCEDIR)/api_reference/_autogen

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
	@rm -rf "$(AUTOGENDIR)"
	@echo "Delete _autogen output"
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# ------------------------------------------------------------
# Catch-all target
# ------------------------------------------------------------
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)