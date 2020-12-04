import numpy

from inPYinting.algorithms.fast_marching.fast_marching_inpainter import FastMarchingInpainter
from inPYinting.algorithms.navier_stokes.navier_stokes_inpainter import NavierStokesInpainter
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
