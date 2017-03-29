class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class InfeasibleError(Error):
    def __init__(self, message):
        self.message = message

class NonLinear(Error):
    def __init__(self, message):
        self.message = message
