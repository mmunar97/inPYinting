import numpy
import pyfftw
import time

from scipy.sparse import spdiags

from inPYinting.algorithms.pde.utils.pde_inpainter import PdeInpainter
from inPYinting.base.result import InpaintingResult


class CahnHilliardInpainter(PdeInpainter):

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
        super(CahnHilliardInpainter, self).__init__(image, mask)

    def inpaint(self, differential_term: float, fidelity: float, max_iterations: int):
        """
        Inpaints the image using the Cahn-Hilliard inpainting.

        References:
            Burger M, He L & Schönlieb (2009)
            Cahn-Hilliard Inpainting and a Generalization for Grayvalue Images.
            SIAM Journal on Image Sciences.

        Args:
            differential_term: A float value, representing the weight given to the new reconstruction in each step.
            fidelity: A float value, representing the weight given to the correct pixels.
            max_iterations: An integer, representing the maximum number of iterations to perform in each channel.

        Returns:
            An InpaintingResult object, representing the result of the inpaint.
        """
        elapsed_time = time.time()
        result = self.__cahn_hilliard_inpainting(differential_term, fidelity, max_iterations)
        elapsed_time = time.time() - elapsed_time

        return InpaintingResult(inpainted_image=self.change_output_format(result), elapsed_time=elapsed_time)

    def __cahn_hilliard_inpainting(self, differential_term: float, fidelity: float, max_iterations: int) -> numpy.ndarray:
        """
        Solves the Cahn-Hilliard inpainting.
        """
        if self.image.ndim == 3:
            M, N, C = self.image.shape
        else:
            M, N = self.image.shape
            C = 1

        hi = 1
        hj = 1

        epsilon = numpy.array([100, 1])

        swap = round(max_iterations/2)
        epsilon1 = numpy.array(epsilon[0]*numpy.ones((swap, 1)))
        epsilon2 = numpy.array(epsilon[1]*numpy.ones((max_iterations-swap, 1)))
        epsilon_list = numpy.concatenate([epsilon1, epsilon2])

        fidelity_array = fidelity * self.mask[:, :, 1]

        l1 = numpy.array(2 * (numpy.cos(2 * numpy.array(range(0, M)) * numpy.pi / M) - 1))
        l2 = numpy.array(2 * (numpy.cos(2 * numpy.array(range(0, N)) * numpy.pi / N) - 1))

        lambda1 = spdiags(l1, 0, M, M)/(hi**2)
        lambda2 = spdiags(l2, 0, N, N)/(hj**2)

        denominator = lambda1.dot(numpy.ones((M, N)))+lambda2.T.dot(numpy.ones((N, M))).T

        u = numpy.ones((M, N, C))

        for channel in range(0, C):

            uu = self.image[:, :, channel]
            u_hat = pyfftw.interfaces.numpy_fft.fft2(uu)
            fidelity_u0_hat = pyfftw.interfaces.numpy_fft.fft2(fidelity_array * uu)

            for iteration in range(0, max_iterations):
                fidelity_u_hat = pyfftw.interfaces.numpy_fft.fft2(fidelity_array * uu)

                f_prime_hat = pyfftw.interfaces.numpy_fft.fft2(2*(2*uu**3 - 3*uu**2 + uu))

                u1 = (1 + differential_term * fidelity_array) * u_hat
                u2 = -(differential_term / epsilon[1]) * denominator * u_hat
                u3 = (differential_term / epsilon_list[iteration]) * denominator * f_prime_hat
                u4 = differential_term * (fidelity_u0_hat - fidelity_u_hat)

                uden = 1.0 + differential_term * (fidelity_array + epsilon_list[iteration] * (denominator ** 2) - (denominator / epsilon[1]))
                u_hat = (u1 + u2 + u3 + u4) / uden

                uu = pyfftw.interfaces.numpy_fft.ifft2(u_hat)

            u[:, :, channel] = uu.real

        return u
