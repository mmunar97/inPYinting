import cv2
import numpy
import time

from inPYinting.algorithms.pde.utils.pde_inpainter import PdeInpainter
from inPYinting.algorithms.pde.utils.utils import get_kernel_derivatives
from inPYinting.base.result import InpaintingResult


class AmleInpainter(PdeInpainter):

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
        super(AmleInpainter, self).__init__(image, mask)

    def inpaint(self, fidelity: float, tolerance: float, max_iterations: int, differential_term: float) -> InpaintingResult:
        """
        Inpaints the image using the Absolute Minimizing Lipschitz Extension inpainting.

        References:
            - Caselles, V., Morel, J. M., & Sbert, C. (1998).
            An axiomatic approach to image interpolation.
            Image Processing, IEEE Transactions on, 7(3), 376-386.

            - Almansa, A. (2002).
            Echantillonnage, interpolation et detection: applications en imagerie satellitaire
            Doctoral dissertation, Cachan, Ecole normale superieure

        Args:
            fidelity: A float value, representing the weight given to the correct pixels.
            tolerance: A float value, representing the threshold to stop the execution.
            max_iterations: An integer, representing the maximum number of iterations to perform in each channel.
            differential_term: A float value, representing the weight given to the new reconstruction in each step.

        Returns:
            An InpaintingResult object, representing the result of the inpaint.
        """
        elapsed_time = time.time()
        result = self.__amle_inpaint(fidelity, tolerance, max_iterations, differential_term)
        elapsed_time = time.time()-elapsed_time

        return InpaintingResult(inpainted_image=self.change_output_format(result), elapsed_time=elapsed_time)

    def __amle_inpaint(self, fidelity: float, tolerance: float, max_iterations: int, differential_term: float) -> numpy.ndarray:
        """
        Applies the AMLE inpainting method to the image.
        """
        if self.image.ndim == 3:
            M, N, C = self.image.shape
        else:
            M, N = self.image.shape
            C = 1

        kfi, kfj, kbi, kbj, kci, kcj = get_kernel_derivatives()

        u = self.image.copy()
        v = numpy.zeros((M, N, 2))

        for channel in range(0, C):
            for _ in range(0, max_iterations):

                # First order derivatives
                ux = cv2.filter2D(u[:, :, channel], -1, kfi)
                uy = cv2.filter2D(u[:, :, channel], -1, kfj)

                # Second order derivatives
                uxx = cv2.filter2D(ux, -1, kbi)
                uxy = cv2.filter2D(ux, -1, kbj)
                uyx = cv2.filter2D(uy, -1, kbi)
                uyy = cv2.filter2D(uy, -1, kbj)

                # Direction field Du/|Du| with central differences
                v[:, :, 0] = cv2.filter2D(u[:, :, channel], -1, kci)
                v[:, :, 1] = cv2.filter2D(u[:, :, channel], -1, kcj)

                norm = numpy.sqrt(numpy.sum(v ** 2, axis=2) + 1e-15)
                v[:, :, 0] = v[:, :, 0] / norm
                v[:, :, 1] = v[:, :, 1] / norm

                # AMLE step
                unew = u[:, :, channel] + differential_term * (uxx * v[:, :, 0] ** 2 + uyy * v[:, :, 1] ** 2 +
                                                               (uxy + uyx) * (v[:, :, 0] * v[:, :, 1]) +
                                                               fidelity * self.mask[:, :, channel] * (self.image[:, :, channel] - u[:, :, channel]))

                similarity = numpy.linalg.norm(unew.reshape(M * N, 1) - u[:, :, channel].reshape(M * N, 1), 2) / numpy.linalg.norm(
                    unew.reshape(M * N, 1), 2)

                u[:, :, channel] = unew

                if similarity < tolerance:
                    break

        return u
