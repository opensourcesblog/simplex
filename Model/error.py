class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class InfeasibleError(Error):
    def __init__(self, message):
        self.message = message

class NonLinear(Error):
    def __init__(self, message):
        self.message = message

class Unsolveable(Error):
    def __init__(self, message):
        self.message = message

class Unbounded(Error):
    def __init__(self, variable):
        Exception.__init__(self,"%s is unbounded" % variable) 

