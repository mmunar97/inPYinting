import numpy
import sys
import time

from typing import List, Tuple

from inPYinting.algorithms.exemplar_based.exemplar_based_utils import *
from inPYinting.base.result import InpaintingResult


class ExemplarBasedInpainter:

    def __init__(self, image, mask):
        self.__image = image
        self.__original_mask = ExemplarBasedInpainter.__reverse_mask(image=mask)
        self.__mask = ExemplarBasedInpainter.__reverse_mask(image=mask)

    def inpaint(self, tau: int = 170, size: int = 3) -> InpaintingResult:
        elapsed_time = time.time()
        result, _ = self.__inpaint(tau, size)
        elapsed_time = time.time() - elapsed_time

        return InpaintingResult(inpainted_image=result, elapsed_time=elapsed_time)

    def inpaint_with_steps(self, tau: int, size: int = 3) -> Tuple[InpaintingResult, List[numpy.ndarray]]:
        elapsed_time = time.time()
        result, steps = self.__inpaint(tau, size)
        elapsed_time = time.time() - elapsed_time

        return InpaintingResult(inpainted_image=result, elapsed_time=elapsed_time), steps

    def __inpaint(self, tau: int, size: int) -> Tuple[numpy.ndarray, List[numpy.ndarray]]:
        omega, confidence = self.__compute_previous_terms(tau)
        source, original = numpy.copy(confidence), numpy.copy(confidence)

        im = numpy.copy(self.__image)

        data = numpy.ndarray(shape=self.__image.shape[:2])

        inpainted_finished = False
        steps: int = 0

        image_steps = []

        while not inpainted_finished:
            steps += 1
            print(f"Inpainting with Exemplar-Based – Step {steps}")

            xsize, ysize = source.shape

            grayscale_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            gradient_x = numpy.float32(cv2.convertScaleAbs(cv2.Scharr(grayscale_image, cv2.CV_32F, 1, 0)))
            gradient_y = numpy.float32(cv2.convertScaleAbs(cv2.Scharr(grayscale_image, cv2.CV_32F, 0, 1)))

            gradient_x[self.__mask == 1] = 0
            gradient_y[self.__mask == 1] = 0
            gradient_x, gradient_y = gradient_x / 255, gradient_y / 255

            d_omega, normal = ExemplarBasedInpainter.__compute_boundary(self.__mask, source)

            confidence, data, index = ExemplarBasedInpainter.__compute_priority(im, size, self.__mask, d_omega, normal,
                                                                                data, gradient_x, gradient_y,
                                                                                confidence)

            list, pp = ExemplarBasedInpainter.__get_patch(d_omega, index, im, original, self.__mask, size)

            im, gradient_x, gradient_y, confidence, source, self.__mask = ExemplarBasedInpainter.__update(im,
                                                                                                          gradient_x,
                                                                                                          gradient_y,
                                                                                                          confidence,
                                                                                                          source,
                                                                                                          self.__mask,
                                                                                                          d_omega, pp,
                                                                                                          list, index,
                                                                                                          size)

            inpainted_finished = True
            for x in range(xsize):
                for y in range(ysize):
                    if source[x, y] == 0:
                        inpainted_finished = False

            image_steps.append(im)

        return im, image_steps

    @staticmethod
    def __reverse_mask(image: numpy.ndarray):
        """
        Converts a mask with black background and white lost pixels into an image with white background and black pixels
        to recover.

        Args:
            image: A two dimensional image, representing the mask with the lost pixels.

        Returns:
            A two dimensional image, representing the mask with the lost pixels marked with black.
        """
        return 255-image

    def __compute_previous_terms(self, tau: int) -> Tuple[List, numpy.ndarray]:
        """
        Computes the list of pixels to be corrected and the matrix of confidence of the image.

        Args:
            tau: A threshold to indicate the limit to modify the mask.

        Returns:
            A tuple of two elements: the list of pixels and the matrix of confidence.
        """
        omega = []
        confidence = numpy.copy(self.__mask)

        for x in range(self.__image.shape[0]):
            for y in range(self.__image.shape[1]):
                mask_value = self.__mask[x, y]
                if mask_value < tau:
                    omega.append([x, y])
                    self.__image[x, y] = [255, 255, 255]
                    self.__mask[x, y] = 1
                    confidence[x, y] = 0
                else:
                    self.__mask[x, y] = 0
                    confidence[x, y] = 1

        return omega, confidence

    @staticmethod
    def __compute_boundary(mask: numpy.ndarray, source: numpy.ndarray):
        """
        Computes the boundary pixels.
        """
        d_omega = []
        normal = []

        laplacian = cv2.filter2D(mask, cv2.CV_32F, get_laplacian_operator())
        gradient_x = cv2.filter2D(source, cv2.CV_32F, get_derivative_x_operator())
        gradient_y = cv2.filter2D(source, cv2.CV_32F, get_derivative_y_operator())

        xsize, ysize = laplacian.shape
        for x in range(xsize):
            for y in range(ysize):
                if laplacian[x, y] > 0:
                    d_omega += [(y, x)]
                    dx = gradient_x[x, y]
                    dy = gradient_y[x, y]

                    norm = (dy ** 2 + dx ** 2) ** 0.5
                    if norm != 0:
                        normal += [(dy / norm, -dx / norm)]
                    else:
                        normal += [(dy, -dx)]

        return d_omega, normal

    @staticmethod
    def __generate_patch_coordinates(image: numpy.ndarray, size: int, point: Tuple[int, int]):
        """
        Computes the extreme points of a patch.
        """
        px, py = point
        xsize, ysize, c = image.shape

        x3 = max(px - size, 0)
        y3 = max(py - size, 0)
        x2 = min(px + size, ysize - 1)
        y2 = min(py + size, xsize - 1)
        return (x3, y3), (x2, y2)

    @staticmethod
    def __compute_confidence(confidence, image, size, mask, d_omega):
        """
        Computes the confidence.
        """
        for k in range(len(d_omega)):
            px, py = d_omega[k]
            patch = ExemplarBasedInpainter.__generate_patch_coordinates(image=image, size=size, point=d_omega[k])
            x3, y3 = patch[0]
            x2, y2 = patch[1]

            i = 0

            size_psi_p = ((x2 - x3 + 1) * (y2 - y3 + 1))

            for x in range(x3, x2 + 1):
                for y in range(y3, y2 + 1):
                    if mask[y, x] == 0:
                        i += confidence[y, x]
            confidence[py, px] = i / size_psi_p
        return confidence

    @staticmethod
    def __compute_data(d_omega, normal, data, gradient_x, gradient_y):
        for k in range(len(d_omega)):
            x, y = d_omega[k]
            n_x, n_y = normal[k]

            data[y, x] = (((gradient_x[y, x] * n_x)**2 + (gradient_y[y, x] * n_y)**2)**0.5) / 255.0
        return data

    @staticmethod
    def __compute_priority(image, size, mask, d_omega, normal, data, gradient_x, gradient_y, confidence):
        conf = ExemplarBasedInpainter.__compute_confidence(confidence, image, size, mask, d_omega)
        dat = ExemplarBasedInpainter.__compute_data(d_omega, normal, data, gradient_x, gradient_y)

        index = 0
        maxi = 0

        for i in range(len(d_omega)):
            x, y = d_omega[i]
            P = conf[y, x] * dat[y, x]
            if P > maxi:
                maxi = P
                index = i

        return conf, dat, index

    @staticmethod
    def __get_patch(d_omega, cible_index, im, original, mask, size):
        mini = minvar = sys.maxsize
        source_patch = []

        p = d_omega[cible_index]
        patch = ExemplarBasedInpainter.__generate_patch_coordinates(im, size, p)

        x1, y1 = patch[0]
        x2, y2 = patch[1]

        x_size, y_size, c = im.shape

        counter, cibles, ciblem, xsize, ysize = ExemplarBasedInpainter.__crible(y2-y1+1, x2-x1+1, x1, y1, mask)
        for x in range(x_size - xsize):
            for y in range(y_size - ysize):
                if ExemplarBasedInpainter.__is_patch_complete(x, y, xsize, ysize, original):
                    source_patch += [(x, y)]
        for (y, x) in source_patch:
            R = V = B = ssd = 0
            for (i, j) in cibles:
                ima = im[y + i, x + j]
                omega = im[y1 + i, x1 + j]
                for k in range(3):
                    difference = float(ima[k]) - float(omega[k])
                    ssd += difference ** 2
                R += ima[0]
                V += ima[1]
                B += ima[2]
            ssd /= counter
            if ssd < mini:
                variation = 0
                for (i, j) in ciblem:
                    ima = im[y + i, x + j]
                    differenceR = ima[0] - R / counter
                    differenceV = ima[1] - V / counter
                    differenceB = ima[2] - B / counter
                    variation += differenceR ** 2 + differenceV ** 2 + differenceB ** 2
                if ssd < mini or variation < minvar:
                    minvar = variation
                    mini = ssd
                    pointPatch = (x, y)
        return ciblem, pointPatch

    @staticmethod
    def __crible(x_size, y_size, x1, y1, mask):
        counter = 0
        cibles, ciblem = [], []
        for i in range(x_size):
            for j in range(y_size):
                if mask[y1 + i, x1 + j] == 0:
                    counter += 1
                    cibles += [(i, j)]
                else:
                    ciblem += [(i, j)]
        return counter, cibles, ciblem, x_size, y_size

    @staticmethod
    def __is_patch_complete(x, y, x_size, y_size, original):
        for i in range(x_size):
            for j in range(y_size):
                if original[x + i, y + j] == 0:
                    return False
        return True

    @staticmethod
    def __update(im, gradient_x, gradient_y, confidence, source, mask, d_omega, point, list, index, size):
        p = d_omega[index]
        patch = ExemplarBasedInpainter.__generate_patch_coordinates(im, size, p)
        x1, y1 = patch[0]
        px, py = point
        for (i, j) in list:
            im[y1 + i, x1 + j] = im[py + i, px + j]
            confidence[y1 + i, x1 + j] = confidence[py, px]
            source[y1 + i, x1 + j] = 1
            mask[y1 + i, x1 + j] = 0

        return im, gradient_x, gradient_y, confidence, source, mask
