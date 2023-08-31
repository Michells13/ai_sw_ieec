import tensorflow as tf
import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K

def mean_iou(y_true, y_pred):
    yt0 = y_true[:,:,:,0]
    yp0 = K.cast(y_pred[:,:,:,0] > 0.5, 'float32')
    inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.math.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1., tf.cast(inter/union, 'float32'))
    return iou
# Load your TensorFlow model
saved_model_dir = "C:/Users/MICHE/Desktop/Master/MTP/tensorFlow/unet.h5"
loaded_model = tf.keras.models.load_model(saved_model_dir, custom_objects={'mean_iou': mean_iou})
#loaded_model = tf.keras.models.load_model(saved_model_dir)

# Optimize the model for TensorFlow Lite conversion
converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)

# Set optimization options
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Specify target platform constraints (if needed)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Function to generate representative dataset
def representative_dataset_gen():
    # Replace with the actual paths to your train/test folders
    data_dirs = ['C:/Users/MICHE/Documents/Datasets/Cloudsen12/split2/train', 'C:/Users/MICHE/Documents/Datasets/Cloudsen12/split2/test']
    num_calibration_images = 100  # Number of images for calibration
    
    for data_dir in data_dirs:
        class_dirs = glob.glob(data_dir + "/*")  # Assuming one subfolder per class
        for class_dir in class_dirs:
            image_paths = glob.glob(class_dir + "/*.jpg")  # Assuming jpg images
            for i, image_path in enumerate(image_paths):
                if i >= num_calibration_images:
                    break
                img = load_img(image_path, target_size=(256, 256))  # Adjust target size as needed
                img_array = img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
                yield [img_array]  # Provide representative input data

# Set the representative dataset function
converter.representative_dataset = representative_dataset_gen

# Convert the model to TensorFlow Lite format
tflite_model = converter.convert()

# Save the optimized TensorFlow Lite model to a file
with open('C:/Users/MICHE/Desktop/tflite_models/optimized_model_unet.tflite', 'wb') as f:
    f.write(tflite_model)


