class BaseSimCLRException(Exception):
    """Base Exception"""

class InvalidBackboneError(BaseSimCLRException):
    """Raised when the choice of backbone Convet is invalid"""

class InvalidDataetSelection(BaseSimCLRException):
    """Raise when the choice of dataset is invalid"""