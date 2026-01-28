import tomllib
from pathlib import Path


def get_version():
    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    if pyproject.exists():
        with pyproject.open("rb") as f:
            data = tomllib.load(f)
        # Adjust depending on your project structure: poetry vs PEP 621
        return data.get("project", {}).get("version")
    return "0.0.0"


__version__ = get_version()
