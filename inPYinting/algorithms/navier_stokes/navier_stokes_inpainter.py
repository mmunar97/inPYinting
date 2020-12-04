import cv2
import numpy
import time

from inPYinting.base.result import InpaintingResult


class NavierStokesInpainter:

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

    def inpaint(self, inpaint_radius: int) -> InpaintingResult:
        """
        Inpaints the image solving the Navier-Stokes equations.

        Reference:
            Bertalmio, Marcelo, Andrea L. Bertozzi, and Guillermo Sapiro. "Navier-stokes, fluid dynamics, and image and
                video inpainting."
            In Computer Vision and Pattern Recognition, 2001. CVPR 2001.
            Proceedings of the 2001 IEEE Computer Society Conference on, vol. 1, pp. I-355. IEEE, 2001.

        Args:
            inpaint_radius: An integer, representing the radius of a circular neighborhood of each point inpainted that
                            is considered by the algorithm.

        Returns:
            An InpaintingResult object, representing the result of the inpaint.
        """
        elapsed_time = time.time()
        result = cv2.inpaint(src=self.__image, inpaintMask=self.__mask,
                             inpaintRadius=inpaint_radius, flags=cv2.INPAINT_NS)
        elapsed_time = time.time()-elapsed_time

        return InpaintingResult(inpainted_image=result, elapsed_time=elapsed_time)
