import tomllib
import pathlib

pyproject_path = pathlib.Path("pyproject.toml")
requirements_path = pathlib.Path("requirements-atompy.txt")

with pyproject_path.open("rb") as f:
    data = tomllib.load(f)

dependencies = data.get("project", {}).get("dependencies", [])

with requirements_path.open("w") as f:
    for dep in dependencies:
        f.write(dep + "\n")

print(f"Wrote {len(dependencies)} dependencies to {requirements_path}")
