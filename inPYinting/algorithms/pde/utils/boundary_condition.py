from enum import Enum


class BoundaryCondition(Enum):
    NEUMANN = "NEUMANN"
    PERIODIC = "PERIODIC"
