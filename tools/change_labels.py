from PIL import Image
import matplotlib.pyplot as plt

def convert_to_labelNom(image):
    new_image = Image.new("L", image.size)
    pixels = image.load()

    # Ancho y alto de la imagen
    width, height = new_image.size

    # Recorrer todos los píxeles y hacer la conversión
    for x in range(width):
        for y in range(height):
            # Si el valor del píxel no es 1, convertirlo a 2
            if pixels[x, y] != 1:
                pixels[x, y] = 2


       

    return image
    


# Ruta de la imagen binaria

im= "C:/Users/MICHE/Documents/Datasets/split_segmentation/annotations/47.png"
image = Image.open(im)
image2=convert_to_labelNom(image)
image2.show()












