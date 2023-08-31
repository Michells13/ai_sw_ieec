import os
import cv2
import numpy as np
import shutil
from PIL import Image
import cv2
from utils_segmentation import createLocations , write_subdataset, convertTif2Jpg

'''

This script is designed to use the output of dd.py and use the extracted 
datataset to put all the images in the structure of tensorflow
 
 Input :-----------------------------------------------------------------
        path_in = path where all the extracted folders are
        path_out= path where the output folder structure will be located 
        split   = percentage of the data for training (the rest will be considered for testing) 
        nf      = Max limit of images
 Output:----------------------------------------------------------------
        Images with the structure of tensorflow for classification in an espesified foler 
 
'''



#Define paths


path_in = "/home/msiau/data/tmp/mvargas/cloudsen2/"
path_out= "/home/msiau/workspace/segmentation/"

limit= 100

#define type of label to use
type_of_label= "manual_hq"

#create folders if they don't exist 
createLocations(path_out)


#check it out how manny subfolders there are
subfolders = [f for f in os.listdir(path_in) if os.path.isdir(os.path.join(path_in, f))]
i=0
#split the data into test and train  by a 0.X factor 

print("Starting process, please wait")
#locate the data in the database structure of tensorflow
for subfolder in subfolders:
    write_subdataset(path_in + subfolder ,path_out , type_of_label, limit)
  
convertTif2Jpg(path_out)