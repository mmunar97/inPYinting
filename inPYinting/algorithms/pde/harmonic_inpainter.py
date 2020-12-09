import cv2
import numpy
import time

from inPYinting.algorithms.pde.utils.pde_inpainter import PdeInpainter
from inPYinting.base.result import InpaintingResult


class HarmonicInpainter(PdeInpainter):

    def __init__(self, image: numpy.ndarray, mask: numpy.ndarray):
        """
        Initializes the object that inpaints a given image with the missing pixels represented by a mask.

        Args:
            image: A three-dimensional numpy array, representing the image to be inpainted which entries are in 0...255
                  range and the channels are BGR.
            mask: A two-dimensional numpy array, representing the mask with missing pixels. The value of each entry in
                  the matrix must be in {0, 255} range; that is, only binary images are allowed. If the pixels needs to
                  be recovered, its value must be 255 and 0 otherwise.
        """
        super(HarmonicInpainter, self).__init__(image, mask)

    def inpaint(self, fidelity: float, tolerance: float, max_iterations: int, differential_term: float) -> InpaintingResult:
        """
        Inpaints the image using the Harmonic method.

        References:
            Shen, J., & Chan, T. F. (2002).
            Mathematical models for local nontexture inpaintings.
            SIAM Journal on Applied Mathematics, 62(3), 1019-1043.

        Args:
            fidelity: A float value, representing the weight given to the correct pixels.
            tolerance: A float value, representing the threshold to stop the execution.
            max_iterations: An integer, representing the maximum number of iterations to perform in each channel.
            differential_term: A float value, representing the weight given to the new reconstruction in each step.

        Returns:
            An InpaintingResult object, representing the result of the inpaint.
        """
        elapsed_time = time.time()
        result = self.__harmonic_inpaint(fidelity, tolerance, max_iterations, differential_term)
        elapsed_time = time.time() - elapsed_time

        return InpaintingResult(inpainted_image=self.change_output_format(result), elapsed_time=elapsed_time)

    def __harmonic_inpaint(self, fidelity: float, tolerance: float, max_iterations: int, differential_term: float) -> numpy.ndarray:
        """
        Applies the Harmonic inpainting method to the image.
        """
        if self.image.ndim == 3:
            M, N, C = self.image.shape
        else:
            M, N = self.image.shape
            C = 1

        u = self.image.copy()

        for channel in range(0, C):
            for _ in range(0, max_iterations):

                laplacian = cv2.Laplacian(u[:, :, channel], cv2.CV_64F)
                unew =  u[:, :, channel] + differential_term*(laplacian + fidelity * self.mask[:, :, channel] * (self.image[:, :, channel]-u[:, :, channel]))

                similarity = numpy.linalg.norm(unew.reshape(M*N, 1)-u[:, :, channel].reshape(M*N, 1), 2)/numpy.linalg.norm(unew.reshape(M*N, 1), 2)

                u[:, :, channel] = unew

                if similarity < tolerance:
                    break

        return u
