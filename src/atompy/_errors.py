from typing import Literal

class NonconstantBinsizeError(Exception):
    def __init__(
        self,
        fname: str,
        which: Literal["x", "y", ""]
    ) -> None:
        self.fname = fname
        self.which = which

    def __str__(self):
        dash = "" if self.which == "" else "-"
        return (
            f"{self.which}binsizes from {self.fname} are not constant and "
            f"no {self.which}{dash}limits are provided."
        )

class UnderdeterminedBinsizeError(Exception):
    def __str__(self) -> str:
        return (
            "Distance between points is not constant and no lower or upper "
            "limit is provided. Provide at least one limit so I can "
            "determine the binsizes."
        )

class AliasError(Exception):
    def __init__(self,
                 keyword_arg: str,
                 alias: str):
        self.keyword_arg = keyword_arg
        self.alias = alias

    def __str__(self):
        return (f"Both '{self.keyword_arg}' and '{self.alias}' have been "
                "provided, but they are aliases")

class FigureWidthTooLargeError(Exception):
    def __str__(self):
        return (
            "New figure width exceeds maximum allowed figure width"
        )
