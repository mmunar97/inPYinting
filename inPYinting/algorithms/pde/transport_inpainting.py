import cv2
import numpy
import time

from inPYinting.algorithms.pde.utils.pde_inpainter import PdeInpainter
from inPYinting.algorithms.pde.utils.utils import get_kernel_derivatives
from inPYinting.base.result import InpaintingResult


class TransportInpainter(PdeInpainter):

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
        super(TransportInpainter, self).__init__(image, mask)

    def inpaint(self, tolerance: float, differential_term: float, epsilon: float, iterations_inpainting: int,
                iterations_anisotropic: int, max_iterations: int) -> InpaintingResult:
        """
        Inpaints the image solving the transport equation.

        References:
            Carola-Bibiane Schönlieb (2015).
            Partial Differential Equation Methods for Image Inpainting.
            University of Cambridge.

        Args:
            tolerance: A float value, representing the threshold to stop the execution.
            differential_term: A float value, representing the weight given to the new reconstruction in each step.
            epsilon: A float value, representing the epsilon constant in the transport equation.
            iterations_inpainting: An integer, representing the number of iterations to perform in the inpainting.
            iterations_anisotropic: An integer, representing the number of iterations to perform while computing
                                    the anisotropic diffusion.
            max_iterations: An integer, representing the maximum number of iterations in global.

        Returns:
            An InpaintingResult object, representing the result of the inpaint.
        """
        elapsed_time = time.time()

        result = self.__transport_inpainting(tolerance, differential_term, epsilon, iterations_inpainting,
                                             iterations_anisotropic, max_iterations)

        elapsed_time = time.time() - elapsed_time

        return InpaintingResult(inpainted_image=self.change_output_format(result), elapsed_time=elapsed_time)

    def __transport_inpainting(self, tolerance: float, differential_term: float, epsilon: float,
                               iterations_inpainting: int,
                               iterations_anisotropic: int, max_iterations: int) -> numpy.ndarray:
        """
        Solves the transport partial differential equation.
        """
        if self.image.ndim == 3:
            M, N, C = self.image.shape
        else:
            M, N = self.image.shape
            C = 1

        kfi, kfj, kbi, kbj, kci, kcj = get_kernel_derivatives()

        laplacian_gepsilon = numpy.zeros((M, N, C))
        differential_laplacian = numpy.zeros((M, N, 2))
        laplacian = numpy.zeros((M, N, C))
        update = numpy.zeros((M, N, C))
        zeros = numpy.zeros((M, N))
        normal = numpy.zeros((M, N, 2))
        lambda_eps = numpy.zeros((M, N, C))

        u = self.image.copy()
        un = self.image.copy()
        mask_eps = 1 - self.mask.copy()
        gepsilon = 1 - self.mask.copy()

        structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))

        for channel in range(0, C):

            lambda_eps[:, :, channel] = cv2.dilate(mask_eps[:, :, channel], structuring_element,
                                                   iterations=1) - mask_eps[:, :, channel]

            for _ in range(0, 5):
                laplacian_gepsilon[:, :, channel] = cv2.filter2D(gepsilon[:, :, channel], -1, kfi - kbi + kfj - kbj)
                gepsilon[:, :, channel] = gepsilon[:, :, channel] + differential_term * laplacian_gepsilon[:, :,
                                                                                        channel] + lambda_eps[:, :,
                                                                                                   channel] * (
                                                      mask_eps[:, :, channel] - gepsilon[:, :, channel])

            u[:, :, channel] = TransportInpainter.__compute_anisotropic_diffusion(u=u[:, :, channel],
                                                                                  differential_term=differential_term,
                                                                                  epsilon=epsilon,
                                                                                  gepsilon=gepsilon[:, :, channel],
                                                                                  iterations=1)

            for _ in range(0, max_iterations):
                for _ in range(0, iterations_inpainting):
                    laplacian[:, :, channel] = cv2.filter2D(u[:, :, channel], -1, kfi - kbi + kfj - kbj)
                    differential_laplacian[:, :, 0] = cv2.filter2D(laplacian[:, :, channel], -1, kci)
                    differential_laplacian[:, :, 1] = cv2.filter2D(laplacian[:, :, channel], -1, kcj)
                    normal[:, :, 0] = cv2.filter2D(u[:, :, channel], -1, kci)
                    normal[:, :, 1] = cv2.filter2D(u[:, :, channel], -1, kcj)

                    norm = numpy.sqrt(numpy.sum(normal ** 2, axis=2) + epsilon)

                    normal[:, :, 0] = numpy.divide(normal[:, :, 0], norm)
                    normal[:, :, 1] = numpy.divide(normal[:, :, 1], norm)

                    beta = -differential_laplacian[:, :, 0] * normal[:, :, 1] + differential_laplacian[:, :,
                                                                                  1] * normal[:, :, 0]
                    beta_positive = beta > 0

                    uxf = cv2.filter2D(u[:, :, channel], -1, kfi)
                    uxb = cv2.filter2D(u[:, :, channel], -1, kbi)
                    uyf = cv2.filter2D(u[:, :, channel], -1, kfj)
                    uyb = cv2.filter2D(u[:, :, channel], -1, kbj)

                    sl1 = numpy.array([numpy.minimum(uxb, zeros), numpy.maximum(uxf, zeros), numpy.minimum(uyb, zeros),
                                       numpy.maximum(uyf, zeros)])
                    sl2 = numpy.array([numpy.maximum(uxb, zeros), numpy.minimum(uxf, zeros), numpy.maximum(uyb, zeros),
                                       numpy.minimum(uyf, zeros)])
                    slopelim1 = numpy.sum(sl1 ** 2, axis=0)
                    slopelim2 = numpy.sum(sl2 ** 2, axis=0)

                    slopelim = beta_positive * numpy.sqrt(slopelim1) + (1 - beta_positive) * numpy.sqrt(slopelim2)
                    update[:, :, channel] = beta * slopelim

                    u[:, :, channel] = u[:, :, channel] + \
                                       differential_term * (1 - self.mask[:, :, channel]) * update[:, :, channel]

                un[:, :, channel] = TransportInpainter.__compute_anisotropic_diffusion(u=u[:, :, channel],
                                                                                       differential_term=differential_term,
                                                                                       epsilon=epsilon,
                                                                                       gepsilon=gepsilon[:, :, channel],
                                                                                       iterations=iterations_anisotropic)

                diff_u = numpy.linalg.norm(un[:, :, channel].reshape(M*N, 1)-u[:, :, channel].reshape(M*N, 1), 2)/numpy.linalg.norm(un[:, :, channel].reshape(M*N, 1), 2)

                u[:, :, channel] = (1-self.mask[:, :, channel])*un[:, :, channel] + self.mask[:, :, channel]*u[:, :, channel]

                if diff_u < tolerance:
                    break

        return u


    @staticmethod
    def __compute_anisotropic_diffusion(u: numpy.ndarray, differential_term: float, epsilon: float,
                                        gepsilon: numpy.ndarray, iterations: int):
        """
        Computes the anisotropic diffusion of an image.
        """
        diffusion = u.copy()
        kfi, kfj, kbi, kbj, kci, kcj = get_kernel_derivatives()

        kii = kfi - kbi
        kjj = kfj - kbj

        ux = cv2.filter2D(diffusion, -1, kci)
        uy = cv2.filter2D(diffusion, -1, kcj)
        uxx = cv2.filter2D(diffusion, -1, kii)
        uyy = cv2.filter2D(diffusion, -1, kjj)
        uxy = cv2.filter2D(cv2.filter2D(diffusion, -1, kci), -1, kcj)

        for _ in range(0, iterations):
            squared_norm_gradient = ux ** 2 + uy ** 2 + epsilon
            diffusion = diffusion + differential_term * gepsilon * (
                        uyy * (ux ** 2) + uxx * (uy ** 2) - 2 * ux * uy * uxy) / squared_norm_gradient

        return diffusion

