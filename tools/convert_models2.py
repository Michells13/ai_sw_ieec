import tensorflow as tf
from keras import backend as K


def mean_iou(y_true, y_pred):
    yt0 = y_true[:,:,:,0]
    yp0 = K.cast(y_pred[:,:,:,0] > 0.5, 'float32')
    inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.math.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1., tf.cast(inter/union, 'float32'))
    return iou
# Load your TensorFlow model
saved_model_dir = "C:/Users/MICHE/Desktop/tflite_models/best_model_unet.h5"
loaded_model = tf.keras.models.load_model(saved_model_dir, custom_objects={'mean_iou': mean_iou})





# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('C:/Users/MICHE/Desktop/tflite_models/converted_NoQ_unet.tflite', 'wb') as f:
    f.write(tflite_model)