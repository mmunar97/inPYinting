import cv2
import numpy


def get_laplacian_operator() -> numpy.ndarray:
    return numpy.array([[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]])


def get_derivative_x_operator() -> numpy.ndarray:
    return numpy.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])


def get_derivative_y_operator() -> numpy.ndarray:
    return numpy.array([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
