import numpy

from skimage.util import img_as_float, img_as_ubyte
from typing import Tuple


class PdeInpainter:

    def __init__(self, image: numpy.ndarray, mask: numpy.ndarray):
        self.__original_image = image
        self.__original_mask = mask

        self.image, self.mask = PdeInpainter.__change_input_format(image, mask)

    @staticmethod
    def __change_input_format(image: numpy.ndarray, mask: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Converts the image from BGR format and UINT8 range into RGB format and [0, 1] range. Furthermore, changes
        the missing values in the image, adding random noise to the lost area.

        Args:
            image: A three-dimensional array, representing the image.
            mask: A two-dimensional array, representing the binary mask.

        Returns:
            A tuple of two elements, representing the modified image and the mask.
        """
        if image.ndim == 3:
            M, N, C = image.shape
        else:
            M, N = image.shape
            C = 1

        input_image = image[:, :, ::-1]
        input_image = img_as_float(input_image)

        binary_mask = numpy.float64(mask == 0)
        if image.ndim == 3 and binary_mask.ndim < 3:
            binary_mask = numpy.repeat(binary_mask[:, :, numpy.newaxis], C, axis=2)

        if C == 1:
            input_image = numpy.expand_dims(input_image, axis=2)
            binary_mask = numpy.expand_dims(binary_mask, axis=2)

        noise = numpy.random.random((M, N, C))

        return binary_mask * input_image + (1 - binary_mask) * noise, binary_mask

    @staticmethod
    def change_output_format(image: numpy.ndarray) -> numpy.ndarray:
        output_image = image[:, :, ::-1]

        output_image[:, :, 0][output_image[:, :, 0] < -1] = -1
        output_image[:, :, 0][output_image[:, :, 0] > 1] = 1
        output_image[:, :, 1][output_image[:, :, 1] < -1] = -1
        output_image[:, :, 1][output_image[:, :, 1] > 1] = 1
        output_image[:, :, 2][output_image[:, :, 2] < -1] = -1
        output_image[:, :, 2][output_image[:, :, 2] > 1] = 1

        output_image = img_as_ubyte(output_image)

        return output_image
