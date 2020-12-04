import numpy


class InpaintingResult:
    """
    Object that stores the results of the inpainting algorithm.
    """

    def __init__(self,
                 inpainted_image: numpy.ndarray,
                 elapsed_time: float):
        self.inpainted_image = inpainted_image
        self.elapsed_time = elapsed_time
