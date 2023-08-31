import os
import cv2
import numpy as np
import shutil
from PIL import Image
import cv2
from utils_segmentation import createLocations , write_subdataset, convertTif2Jpg, loadmask, convert_to_labelNom, tiffLabel
import rasterio
annotation_path= "C:/Users/MICHE/Desktop/manual_hq.tif"

mask=loadmask(annotation_path)
img = rasterio.open(annotation_path)
img= tiffLabel(img)
img = Image.fromarray(np.uint8(img))
#img.show()
# Save label at "a
# Convert to label nomenclature
img=convert_to_labelNom(img)
#img.show()
# save the image using the save() method
#mask_image.show()
img.save("hola1.PNG", format="PNG")



