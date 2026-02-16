class UnmatchingEdgesError(Exception):
    """Exception for when edges of two histograms do not match"""

    def __init__(self, which):
        self.message = f"histogram {which}edges don't match"
        super().__init__(self.message)
