import os
import cv2
import numpy as np
import shutil
from PIL import Image
import cv2
from rasterio.plot import show
import matplotlib.pyplot as plt
import rasterio
from PIL import Image
import matplotlib.pyplot as plt



def convert_to_labelNom(image):
    # Convertir a escala de grises
    image = image.convert("L")

    # Crear una nueva imagen en blanco con el mismo tamaño que la original
    new_image = Image.new("L", image.size)

    # Obtener los píxeles de ambas imágenes
    pixels = image.load()
    new_pixels = new_image.load()

    # Ancho y alto de la imagen
    width, height = image.size

    # Recorrer todos los píxeles y hacer la conversión
    for x in range(width):
        for y in range(height):
            # Si el valor del píxel no es 1, convertirlo a 2
            if pixels[x, y] != 1:
                new_pixels[x, y] = 2
            else:
                new_pixels[x, y] = pixels[x, y]

    return new_image

def removeTar(path):
    '''
        This function removes a file within the indicated path 

    Parameters
    ----------
    path  : file to be removed

    Returns
    -------
    none:    prints success or failed
    
    
    
    '''
    try:
        os.remove(path)
        print("File removed successfully.")
    except FileNotFoundError:
        print("File not found.")
    except PermissionError:
        print("Permission denied. Unable to remove the file.")
    except Exception as e:
        print("An error occurred:", str(e))
        
        
        
        
        
        
def tiffLabel(img):
    '''
    This function takes a .tif label from cloudsen12 dataset and convert it 
    to a binary image only taking into account the espesified label

    Parameters
    ----------
    img : .tif image label from cloudsen12

    Returns
    -------
    thresholded image (Binary)

    '''
    # Read the image data, one band 
    image = img.read(1)
    #threshold here
    thresholded_image = image.astype(np.uint8)
    return thresholded_image
def crop_image(img):
    height, width, _ = img.shape
    # Calculate the coordinates for cropping
    center_x = width // 2
    center_y = height // 2
    left = center_x - 254
    right = center_x + 254
    top = center_y - 254
    bottom = center_y + 254
    image = img[top:bottom+1, left:right+1]
    return image
    
def convert_images(folder_path):
    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.tif') or file_name.endswith('.tiff'):
            # Generate the file paths
            tiff_path = os.path.join(folder_path, file_name)
            jpg_path = os.path.join(folder_path, file_name.replace('.tif', '.jpg').replace('.tiff', '.jpg'))

            # Read the TIFF image
            tiff_image = cv2.imread(tiff_path)
            tiff_image = crop_image(tiff_image)

            # Write the image as JPG
            cv2.imwrite(jpg_path, tiff_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            # Delete the TIFF image
            os.remove(tiff_path)



def count_files_in_folder(folder_path):
    file_count = 0

    for _, _, files in os.walk(folder_path):
        file_count += len(files)

    return file_count
def copy_and_rename_image(source_path, destination_folder, new_filename):
    if os.path.exists(source_path):
        # Get the file extension from the source path
        _, extension = os.path.splitext(source_path)
        
        # Construct the new destination path with the renamed file
        destination_path = os.path.join(destination_folder, new_filename + extension)
        
        # Copy the image file to the destination folder with the new name
        shutil.copy2(source_path, destination_path)
        #convert_tif_to_png(source_path, destination_folder, new_filename)
        #print("Image copied and renamed successfully.")
    else:
        print("Source image does not exist.")
def copy_image(source_path, destination_path):
    if os.path.exists(source_path):
        shutil.copy2(source_path, destination_path)
        #print("Image copied successfully.")
    else:
        print("Source image does not exist.")
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Folder created:", path)
    else:
        print("Folder exists:", path)


def loadmask(path):
    #img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = rasterio.open(path)
    img= tiffLabel(img)
    if img is not None:
        return img
    else:
        print("Failed to read the image:", path)
        return None
    #print(white_percentage)

def write_subdataset(path,path_out,annotationType,limit):
    
    
    
    subfolders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
  
    if count_files_in_folder(path_out + "/images/") < limit:
        for subfolder in subfolders:
            subfolder_path = os.path.join(path, subfolder)
            subsubfolders= [f for f in os.listdir(subfolder_path) if os.path.isdir(os.path.join(subfolder_path, f))]
    
            for subsubfolder in subsubfolders:
                subsubfolder_path = os.path.join(subfolder_path, subsubfolder)
                annotation_path= subsubfolder_path + "/labels/" + annotationType + ".tif"
                thumbnail_path=  subsubfolder_path + "/" + "thumbnail.tif"
                mask=loadmask(annotation_path)
                
                # Save label at "annotation" location
                mask_image = Image.fromarray(np.uint8(mask))  # Assuming mask is in uint8 format, adjust as needed
                # Convert to label nomenclature
                mask_image=convert_to_labelNom(mask_image)
                # save the image using the save() method
                mask_image.save(path_out + "/annotations/" + str(count_files_in_folder(path_out + "/annotations/")) + ".png", optimize=True)
                copy_and_rename_image(thumbnail_path, path_out + "/images/", str(count_files_in_folder(path_out + "/images/")))
            
def createLocations(path_out):
    create_folder(path_out + "images")
    create_folder(path_out + "annotations")


def convertTif2Jpg(path_out):
    convert_images(path_out + "images")
    convert_images(path_out + "annotations")

    