class UnmatchingEdgesError(Exception):
    """Exception for when edges of two histograms do not match"""

    def __init__(self, which):
        self.message = f"histogram {which}edges don't match"
        super().__init__(self.message)


class AliasError(Exception):
    def __init__(self, keyword_arg: str, alias: str):
        self.keyword_arg = keyword_arg
        self.alias = alias

    def __str__(self):
        return (
            f"Both '{self.keyword_arg}' and '{self.alias}' have been "
            "provided, but they are aliases"
        )
