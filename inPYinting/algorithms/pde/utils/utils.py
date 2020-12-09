import numpy

from typing import Tuple


def get_kernel_derivatives() -> Tuple[numpy.ndarray, numpy.ndarray,
                                      numpy.ndarray, numpy.ndarray,
                                      numpy.ndarray, numpy.ndarray]:
    """
    Returns the kernels that will be used to compute the discrete derivatives.
    """
    forward_kernel_i = numpy.zeros((3, 3))
    forward_kernel_i[1, 1] = -1
    forward_kernel_i[2, 1] = 1

    forward_kernel_j = numpy.zeros((3, 3))
    forward_kernel_j[1, 1] = -1
    forward_kernel_j[1, 2] = 1

    backward_kernel_i = numpy.zeros((3, 3))
    backward_kernel_i[1, 1] = 1
    backward_kernel_i[0, 1] = -1

    backward_kernel_j = numpy.zeros((3, 3))
    backward_kernel_j[1, 1] = 1
    backward_kernel_j[1, 0] = -1

    centered_kernel_i = numpy.zeros((3, 3))
    centered_kernel_i[0, 1] = -0.5
    centered_kernel_i[2, 1] = 0.5

    centered_kernel_j = numpy.zeros((3, 3))
    centered_kernel_j[1, 2] = 0.5
    centered_kernel_j[1, 0] = -0.5

    return forward_kernel_i, forward_kernel_j, backward_kernel_i, backward_kernel_j, centered_kernel_i, centered_kernel_j
