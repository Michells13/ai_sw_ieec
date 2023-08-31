from PIL import Image
import os

input_folder = 'C:/Users/MICHE/Desktop/Master/MTP/tensorFlow/inference'
output_folder = 'C:/Users/MICHE/Desktop/Master/MTP/tensorFlow/inference_qemu'

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]

for image_file in image_files:
    input_path = os.path.join(input_folder, image_file)
    output_path = os.path.join(output_folder, os.path.splitext(image_file)[0] + '.jpeg')

    # Open the image
    img = Image.open(input_path)

    # Resize the image to 224x224
    img_resized = img.resize((256, 256), Image.ANTIALIAS)

    # Save the resized image in JPEG format
    img_resized.save(output_path, 'JPEG')

    print(f'Resized and saved: {output_path}')
