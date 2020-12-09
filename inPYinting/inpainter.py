import numpy

from skimage.morphology import disk

from inPYinting.algorithms.exemplar_based.exemplar_based_inpainting import ExemplarBasedInpainter
from inPYinting.algorithms.fast_marching.fast_marching_inpainter import FastMarchingInpainter
from inPYinting.algorithms.navier_stokes.navier_stokes_inpainter import NavierStokesInpainter
from inPYinting.algorithms.pde.amle_inpainter import AmleInpainter
from inPYinting.algorithms.pde.harmonic_inpainter import HarmonicInpainter
from inPYinting.algorithms.pde.mumford_shah_inpainter import MumfordShahInpainter
from inPYinting.algorithms.softcolor.softcolor_inpainter import SoftcolorInpainter
from inPYinting.base.inpainting_algorithms import InpaintingAlgorithm
from inPYinting.base.result import InpaintingResult


class Inpainter:

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
        self.__image = image
        self.__mask = mask

    def inpaint(self, method: InpaintingAlgorithm, **kwargs) -> InpaintingResult:
        """
        Inpaints the image with a certain method.

        Args:
            method: An InpaintingAlgorithm value, representing the method to be used.
            **kwargs: The parameters of each method.

        Returns:
            An InpaintingResult object, containing the recovered image and the elapsed time.
        """
        if method == InpaintingAlgorithm.FAST_MARCHING:
            return self.__inpaint_with_fast_marching(**kwargs)
        elif method == InpaintingAlgorithm.NAVIER_STOKES:
            return self.__inpaint_with_navier_stokes(**kwargs)
        elif method == InpaintingAlgorithm.SOFTCOLOR_FUZZY_MORPHOLOGY:
            return self.__inpaint_with_softcolor(**kwargs)
        elif method == InpaintingAlgorithm.EXEMPLAR_BASED:
            return self.__inpaint_with_exemplar_based(**kwargs)
        elif method == InpaintingAlgorithm.PDE_AMLE:
            return self.__inpaint_with_pde_amle(**kwargs)
        elif method == InpaintingAlgorithm.PDE_HARMONIC:
            return self.__inpaint_with_pde_harmonic(**kwargs)
        elif method == InpaintingAlgorithm.PDE_MUMFORD_SHAH:
            return self.__inpaint_with_pde_mumford_shah(**kwargs)

    def __inpaint_with_fast_marching(self, **kwargs) -> InpaintingResult:
        """
        Inpaints the image using the Fast-Marching method.
        """
        # PARAMETER EXTRACTION
        inpaint_radius = kwargs.get("inpaint_radius", 20)
        # END

        fast_marching_inpainter = FastMarchingInpainter(image=self.__image, mask=self.__mask)
        return fast_marching_inpainter.inpaint(inpaint_radius)

    def __inpaint_with_navier_stokes(self, **kwargs) -> InpaintingResult:
        """
        Inpaints the image solving the Navier-Stokes equations.
        """
        # PARAMETER EXTRACTION
        inpaint_radius = kwargs.get("inpaint_radius", 20)
        # END

        navier_stokes_inpainter = NavierStokesInpainter(image=self.__image, mask=self.__mask)
        return navier_stokes_inpainter.inpaint(inpaint_radius)

    def __inpaint_with_softcolor(self, **kwargs) -> InpaintingResult:
        """
        Inpaints the image with the softcolor framework.
        """
        # PARAMETER EXTRACTION
        structuring_element = kwargs.get("structuring_element", disk(5).astype('float64'))
        structuring_element[structuring_element == 0] = numpy.nan
        max_iterations = kwargs.get("max_iterations", 10)
        # END

        softcolor_inpainter = SoftcolorInpainter(image=self.__image, mask=self.__mask)
        return softcolor_inpainter.inpaint(structuring_element=structuring_element, max_iterations=max_iterations)

    def __inpaint_with_exemplar_based(self, **kwargs) -> InpaintingResult:
        """
        Inpaints the image with the Exemplar-Based method.
        """
        # PARAMETER EXTRACTION
        tau = kwargs.get("tau", 170)
        size = kwargs.get("size", 3)
        # END

        exemplar_based_inpainter = ExemplarBasedInpainter(self.__image, self.__mask)
        return exemplar_based_inpainter.inpaint(tau=tau, size=size)

    def __inpaint_with_pde_amle(self, **kwargs) -> InpaintingResult:
        """
        Inpaints the image with PDE-AMLE.
        """
        # PARAMETER EXTRACTION
        fidelity = kwargs.get("fidelity", 10**2)
        tolerance = kwargs.get("tolerance", 1e-12)
        max_iterations = kwargs.get("max_iterations", 400)
        differential_term = kwargs.get("differential_term", 0.01)
        # END

        amle_inpainter = AmleInpainter(self.__image, self.__mask)
        return amle_inpainter.inpaint(fidelity, tolerance, max_iterations, differential_term)

    def __inpaint_with_pde_harmonic(self, **kwargs) -> InpaintingResult:
        """
        Inpaints the image with PDE-Harmonic.
        """
        # PARAMETER EXTRACTION
        fidelity = kwargs.get("fidelity", 10)
        tolerance = kwargs.get("tolerance", 1e-5)
        max_iterations = kwargs.get("max_iterations", 500)
        differential_term = kwargs.get("differential_term", 0.1)
        # END

        harominc_inpainter = HarmonicInpainter(self.__image, self.__mask)
        return harominc_inpainter.inpaint(fidelity, tolerance, max_iterations, differential_term)

    def __inpaint_with_pde_mumford_shah(self, **kwargs) -> InpaintingResult:
        """
        Inpaints the image with PDE-Mumford Shah.
        """
        # PARAMETER EXTRACTION
        fidelity = kwargs.get("fidelity", 10e9)
        alpha = kwargs.get("alpha", 1)
        gamma = kwargs.get("gamma", 0.5)
        epsilon = kwargs.get("epsilon", 0.05)
        max_iterations = kwargs.get("max_iterations", 20)
        tolerance = kwargs.get("tolerance", 1e-14)
        # END

        mumford_shah_inpainter = MumfordShahInpainter(self.__image, self.__mask)
        return mumford_shah_inpainter.inpaint(fidelity, alpha, gamma, epsilon, max_iterations, tolerance)
