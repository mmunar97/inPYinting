from enum import Enum


class DerivativeDirection(Enum):
    BACKWARD = "BACKWARD"
    CENTRAL = "CENTRAL"
    FORWARD = "FORWARD"
