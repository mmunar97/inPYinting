import numpy
import time

from inPYinting.algorithms.pde.utils.boundary_condition import BoundaryCondition
from inPYinting.algorithms.pde.utils.derivative_direction import DerivativeDirection
from inPYinting.algorithms.pde.utils.pde_inpainter import PdeInpainter
from inPYinting.base.result import InpaintingResult

from scipy.sparse import eye, identity, kron, spdiags
from scipy.sparse.linalg import spsolve
from scipy.linalg import toeplitz
from typing import Tuple


class MumfordShahInpainter(PdeInpainter):

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
        super(MumfordShahInpainter, self).__init__(image, mask)

    def inpaint(self, fidelity: float, alpha: float, gamma: float, epsilon: float,
                max_iterations: int, tolerance: float) -> InpaintingResult:
        """

        Args:
            fidelity:
            alpha:
            gamma:
            epsilon:
            max_iterations:
            tolerance:

        Returns:

        """
        elapsed_time = time.time()
        result = self.__ms_inpainting(fidelity, alpha, gamma, epsilon, max_iterations, tolerance)
        elapsed_time = time.time() - elapsed_time

        return InpaintingResult(inpainted_image=self.change_output_format(result), elapsed_time=elapsed_time)

    def __ms_inpainting(self, fidelity: float, alpha: float, gamma: float, epsilon: float, max_iterations: int, tolerance: float) -> numpy.ndarray:
        """
        Solves the Mumford-Shah functional equation.
        """
        if self.image.ndim == 3:
            M, N, C = self.image.shape
        else:
            M, N = self.image.shape
            C = 1

        u = self.image.copy().reshape(M*N, C)
        chi = self.mask.copy().reshape(M*N, C)
        fidelity_matrix = fidelity*chi

        rhsl = (fidelity_matrix/gamma)*u
        rhsm = numpy.ones((M*N, C))

        hi = 1
        hj = 1

        Dfi, Dfj = MumfordShahInpainter.__compute_gradient(image=self.image[:, :, 0], hi=hi, hj=hj,
                                                           direction=DerivativeDirection.FORWARD,
                                                           boundary=BoundaryCondition.PERIODIC)
        Dbi, Dbj = MumfordShahInpainter.__compute_gradient(image=self.image[:, :, 0], hi=hi, hj=hj,
                                                           direction=DerivativeDirection.BACKWARD,
                                                           boundary=BoundaryCondition.PERIODIC)

        Dci = (Dfi + Dbi) / 2
        Dcj = (Dfj + Dbj) / 2

        laplacian = MumfordShahInpainter.__compute_laplacian(self.image[:, :, 0], hi=hi, hj=hj)

        for channel in range(0, C):
            for _ in range(0, max_iterations):
                chi_matrix = MumfordShahInpainter.__compute_matrix_m(u[:, channel], alpha, gamma, epsilon, Dci, Dcj, laplacian)
                chi_new = spsolve(chi_matrix, rhsm[:, channel])
                chi[:, channel] = chi_new

                u_matrix = MumfordShahInpainter.__compute_matrix_l(chi[:, channel], fidelity_matrix[:,channel],
                                                                   gamma, epsilon, Dci, Dcj, laplacian)
                u_new = spsolve(u_matrix, rhsl[:, channel])

                similarity = numpy.linalg.norm(u_new-u[:, channel], 2)/numpy.linalg.norm(u_new, 2)
                u[:, channel] = u_new

                if similarity < tolerance:
                    break

        return u.reshape(M, N, C)

    @staticmethod
    def __compute_gradient(image: numpy.ndarray,
                           hi: int, hj: int,
                           direction: DerivativeDirection,
                           boundary: BoundaryCondition) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Computes the gradient of a two dimensional image.

        Args:
            image: A two-dimensional numpy array, representing the image.
            hi: An integer, representing the width of the step in the horizontal direction.
            hj: An integer, representing the width of the step in the vertical direction.
            direction: A DerivativeDirection value, representing which derivative is used.
            boundary: A BoundaryCondition value, representing the boundary conditions.

        Returns:
            A tuple of two numpy arrays, representing the gradient in each direction.
        """
        M, N = image.shape

        D1 = None
        D2 = None

        if direction == DerivativeDirection.BACKWARD:
            D1 = spdiags([numpy.ones(M), -numpy.ones(M)], [-1, 0], M, M) / hi
            D2 = spdiags([numpy.ones(N), -numpy.ones(N)], [-1, 0], N, N) / hj
        elif direction == DerivativeDirection.CENTRAL:
            D1 = spdiags([-numpy.ones(M), numpy.ones(M)], [-1, 1], M, M) / (2 * hi)
            D2 = spdiags([-numpy.ones(N), numpy.ones(N)], [-1, 1], N, N) / (2 * hj)
        elif direction == DerivativeDirection.FORWARD:
            D1 = spdiags([-numpy.ones(M), numpy.ones(M)], [0, 1], M, M) / hi
            D2 = spdiags([-numpy.ones(N), numpy.ones(N)], [0, 1], N, N) / hj

        D1 = D1.tolil()
        D2 = D2.tolil()

        if boundary == BoundaryCondition.NEUMANN:
            if direction == DerivativeDirection.BACKWARD:
                D1[0, :] = 0
                D2[0, :] = 0
            elif direction == DerivativeDirection.CENTRAL:
                D1[(0, M - 1), :] = 0
                D2[(0, M - 1), :] = 0
            elif direction == DerivativeDirection.FORWARD:
                D1[M - 1, :] = 0
                D2[N - 1, :] = 0
        elif boundary == BoundaryCondition.PERIODIC:
            if direction == DerivativeDirection.FORWARD:
                D1[0, 1] = 0
                D2[0, 1] = 0
                D1[M - 1, M - 2] = 0
                D2[N - 1, N - 2] = 0
                D1[0, M - 1] = -1
                D2[0, N - 1] = -1
                D1[M - 1, 0] = 1
                D2[N - 1, 0] = 1
            if direction == DerivativeDirection.BACKWARD:
                D1[0, 1] = 0
                D2[0, 1] = 0
                D1[M - 1, M - 2] = 0
                D2[N - 1, N - 2] = 0
                D1[0, M - 1] = 1
                D2[0, N - 1] = 1
                D1[M - 1, 0] = -1
                D2[N - 1, 0] = -1

        D1 = kron(D1, identity(N))
        D2 = kron(identity(M), D2)

        return D1, D2

    @staticmethod
    def __compute_divergence(vector_field_i: numpy.ndarray, vector_field_j: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Computes the divergence operator of each vector field. The divergence of a vector field is defined as minus
        the transposed operator.

        Args:
            vector_field_i: A numpy array, representing one vector field.
            vector_field_j: A numpy array, representing one vector field.

        Returns:
            A tuple of two elements, representing the divergence of each vector field.
        """
        return -vector_field_i.T, -vector_field_j.T

    @staticmethod
    def __compute_laplacian(image: numpy.ndarray, hi: int, hj: int) -> numpy.ndarray:
        """
        Computes the laplacian of an image.

        Args:
            image: A two-dimensional numpy array, representing the image.
            hi: An integer, representing the width of the step in the horizontal direction.
            hj: An integer, representing the width of the step in the vertical direction.

        Returns:
            A numpy array, representing the laplacian of the image.
        """
        M, N = image.shape

        D2i = toeplitz(numpy.append(numpy.array([-2, 1]), numpy.zeros((M-2, 1))))/(hi**2)
        D2j = toeplitz(numpy.append(numpy.array([-2, 1]), numpy.zeros((N-2, 1))))/(hj**2)

        D2i[0, 1] = 2 / (hi ** 2)
        D2i[M - 2, M - 1] = 2 / (hi ** 2)
        D2j[0, 1] = 2 / (hj ** 2)
        D2j[N - 2, N - 1] = 2 / (hj ** 2)

        laplacian = kron(D2i, identity(N)) + kron(identity(M), D2j)

        return laplacian

    @staticmethod
    def __compute_matrix_m(u, alpha, gamma, epsilon, Dic, Djc, L):

        N, = u.shape

        nabla_u_squared = Dic.dot(u.ravel())**2 + Djc.dot(u.ravel())**2
        matrix = eye(N, N) + (2*epsilon*gamma/alpha)*spdiags(nabla_u_squared, 0, N, N) - 4*(epsilon*2)*L

        return matrix

    @staticmethod
    def __compute_matrix_l(chi,FIDELITY,gamma,epsilon,Dic,Djc,L):
        N, = chi.shape

        # Definition of the nonlinear diffusion weighted by \chi^2:
        z = chi ** 2 + epsilon ** 2  # coefficient of nonlinear diffusion

        zx = Dic.dot(z)
        zy = Djc.dot(z)

        Z = spdiags(z, 0, N, N)
        Zx = spdiags(zx, 0, N, N)
        Zy = spdiags(zy, 0, N, N)

        NonlinearDelta = Z.dot(L) + Zx.dot(Dic) + Zy.dot(Djc)

        L = -NonlinearDelta + spdiags(FIDELITY / gamma, 0, N, N)

        return L
