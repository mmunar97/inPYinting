from softcolor.morphology import MorphologyInCIELab
from skimage.util import img_as_float, img_as_ubyte

import numpy
import time

from inPYinting.base.result import InpaintingResult


class SoftcolorInpainter:

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
        self.__image = img_as_float(image[:, :, ::-1])
        self.__mask = mask

    def inpaint(self, structuring_element: numpy.ndarray, max_iterations) -> InpaintingResult:
        """
        Inpaints the image using the softcolor framework, the fuzzy extension of mathematical morphology to
        multivariate images.

        References:
            Bibiloni, P., González-Hidalgo, M., & Massanet, S. (2018).
            Soft Color Morphology: A Fuzzy Approach for Multivariate Images.
            Journal of Mathematical Imaging and Vision, 1-17.

        Args:
            structuring_element: A numpy array, representing the structuring element to be used in the softcolor
                                 erosion and softcolor dilation.
            max_iterations: The maximum number of iterations to do in the recovering.

        Returns:
            An InpaintingResult object, representing the result of the inpaint.
        """
        elapsed_time = time.time()

        morphology = MorphologyInCIELab()

        inpainting_result = morphology.inpaint(multivariate_image=self.__generate_masked_image(),
                                               structuring_element=structuring_element,
                                               max_iterations=max_iterations)
        elapsed_time = time.time() - elapsed_time

        return InpaintingResult(inpainted_image=img_as_ubyte(inpainting_result)[:, :, ::-1], elapsed_time=elapsed_time)

    def __generate_masked_image(self) -> numpy.ndarray:
        """
        Modifies the original image so that it implicitly contains the mask; that is, all the pixels that are marked in
        the mask as lost are marked in the original image as NaN.

        Returns:
            A numpy array, representing the modified image.
        """
        modified_image = self.__image.copy()

        for channel_index in range(modified_image.shape[2]):
            channel = modified_image[:, :, channel_index]
            channel[self.__mask == 255] = numpy.nan
            modified_image[:, :, channel_index] = channel

        return modified_image
