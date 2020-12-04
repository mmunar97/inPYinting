from enum import Enum


class InpaintingAlgorithm(Enum):
    FAST_MARCHING = 0
    NAVIER_STOKES = 1
    SOFTCOLOR_FUZZY_MORPHOLOGY = 2
