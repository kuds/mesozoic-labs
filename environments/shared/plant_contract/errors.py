"""Exceptions raised by the plant contract."""


class PlantContractError(ValueError):
    """Raised when a plant manifest is missing, stale, or internally invalid."""


class PlantCompatibilityError(PlantContractError):
    """Raised when a checkpoint was created for an incompatible plant."""
