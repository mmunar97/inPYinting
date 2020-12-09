import cv2

from inPYinting.base.inpainting_algorithms import InpaintingAlgorithm
from inPYinting.inpainter import Inpainter

from skimage.io import imread

if __name__ == "__main__":

    image = cv2.imread("/Users/marcmunar/Downloads/mumford_shah_input.png")
    mask = 255-cv2.imread("/Users/marcmunar/Downloads/mumford_shah_mask.png", cv2.IMREAD_GRAYSCALE)

    inpainter = Inpainter(image=image, mask=mask)
    #result_fm = inpainter.inpaint(InpaintingAlgorithm.FAST_MARCHING)
    #result_ns = inpainter.inpaint(InpaintingAlgorithm.NAVIER_STOKES)
    #result_sc = inpainter.inpaint(InpaintingAlgorithm.SOFTCOLOR_FUZZY_MORPHOLOGY)
    #result_eb = inpainter.inpaint(InpaintingAlgorithm.EXEMPLAR_BASED)
    #result_pde_amle = inpainter.inpaint(InpaintingAlgorithm.PDE_AMLE)
    result_pde_har = inpainter.inpaint(InpaintingAlgorithm.PDE_HARMONIC)

    cv2.imwrite("/Users/marcmunar/Desktop/inpaint.png", result_pde_har.inpainted_image)
