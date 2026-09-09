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
	@for /d /r "$(SOURCEDIR)/api_reference" %%D in (_autogen) do @rmdir /s /q "%%D"
	@echo "Delete _autogen output"
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# ------------------------------------------------------------
# Catch-all target
# ------------------------------------------------------------
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
