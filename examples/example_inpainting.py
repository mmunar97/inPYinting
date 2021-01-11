import cv2

from inPYinting.base.inpainting_algorithms import InpaintingAlgorithm
from inPYinting.inpainter import Inpainter


if __name__ == "__main__":

    image_path = r"C:\Users\Usuario\Desktop\Inpainting Demo\mumford_shah_clean.png"
    mask_path = r"C:\Users\Usuario\Desktop\Inpainting Demo\mumford_shah_mask.png"

    image = cv2.imread(image_path)
    mask = 255-cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    inpainter = Inpainter(image=image, mask=mask)
    #result_fm = inpainter.inpaint(InpaintingAlgorithm.FAST_MARCHING)
    result_ns = inpainter.inpaint(InpaintingAlgorithm.NAVIER_STOKES)
    #result_sc = inpainter.inpaint(InpaintingAlgorithm.SOFTCOLOR_FUZZY_MORPHOLOGY)
    #result_eb = inpainter.inpaint(InpaintingAlgorithm.EXEMPLAR_BASED)
    #result_pde_amle = inpainter.inpaint(InpaintingAlgorithm.PDE_AMLE)
    #result_pde_har = inpainter.inpaint(InpaintingAlgorithm.PDE_HARMONIC)
    #result_pde_ms = inpainter.inpaint(InpaintingAlgorithm.PDE_MUMFORD_SHAH)
    #result_pde_ch = inpainter.inpaint(InpaintingAlgorithm.PDE_CAHN_HILLIARD)
    #result_pde_tr = inpainter.inpaint(InpaintingAlgorithm.PDE_TRANSPORT)

    cv2.imwrite(r"C:\Users\Usuario\Desktop\inpaint.png", result_ns.inpainted_image)
