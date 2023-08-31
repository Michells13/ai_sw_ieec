from PIL import Image
import os

def is_image_file(filename):
    # Verificar si la extensión del archivo corresponde a una imagen compatible
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    return any(filename.lower().endswith(ext) for ext in valid_extensions)

def examinar_imagen(ruta_imagen):
    # Verificar si el archivo es una imagen antes de intentar cargarlo
    if not is_image_file(ruta_imagen):
        print(f"El archivo '{ruta_imagen}' no es una imagen válida.")
        return

    # Cargar la imagen desde la ruta proporcionada
    imagen = Image.open(ruta_imagen)

    # Mostrar algunas propiedades de la imagen
    print("Ancho de la imagen:", imagen.width)
    print("Alto de la imagen:", imagen.height)
    print("Modo de la imagen:", imagen.mode)
    print("Formato de la imagen:", imagen.format)

    # Mostrar la imagen utilizando el visor de imágenes predeterminado
    imagen.show()

    # Retorna el objeto de la imagen para usarlo más adelante si es necesario
    return imagen

# Ruta de la imagen que deseas cargar y examinar
ruta_imagen = "C:/Users/MICHE/Documents/Datasets/unet/annotations/trimaps/american_bulldog_85.png"
ruta_imagen1 = "C:/Users/MICHE/Documents/Datasets/unet/images/american_bulldog_85.jpg"

# Llamar a la función para cargar y examinar la imagen
imagen_cargada = examinar_imagen(ruta_imagen1)
imagen_cargada = examinar_imagen(ruta_imagen)



