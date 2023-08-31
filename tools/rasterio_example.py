import rasterio
from rasterio.plot import show
import numpy as np
import cv2
import matplotlib.pyplot as plt
fp = r'C:/Users/MICHE/Documents/Datasets/Cloudsen12/subset/03__ROI_0047__ROI_0069/ROI_0051/20200215T082021_20200215T084629_T34JCP/labels/manual_hq.tif'
img = rasterio.open(fp)
def tiffLabel(img):
    """
    This function takes a .tif label from cloudsen12 and convert it 
    to a binary image only taking into account the espesified label

    Parameters
    ----------
    img : .tif image label from cloudsen12

    Returns
    -------
    thresholded image (Binary)

    """
    # Read the image data
    image = img.read(1)
    # Apply thresholding
    # Set the threshold range
    lower_threshold = 0  # Set your lower threshold value
    upper_threshold = 3  # Set your upper threshold value
    # Apply thresholding
    thresholded_image = np.where((image > lower_threshold) & (image < upper_threshold), 3, 0)
    # Convert the thresholded image to uint8
    thresholded_image = thresholded_image.astype(np.uint8)
    return thresholded_image
    
    

# # Display the thresholded image
# show(thresholded_image)

# # Close the image dataset
# img.close()