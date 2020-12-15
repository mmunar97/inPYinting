import numpy
import time

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

        Args:
            differential_term:
            fidelity:
            max_iterations:

        Returns:

        """
        elapsed_time = time.time()
        result = self.__cahn_hilliard_inpainting(differential_term, fidelity, max_iterations)
        elapsed_time = time.time() - elapsed_time

        return InpaintingResult(inpainted_image=self.change_output_format(result), elapsed_time=elapsed_time)

    def __cahn_hilliard_inpainting(self, differential_term: float, fidelity: float, max_iterations: int) -> numpy.ndarray:
        pass
