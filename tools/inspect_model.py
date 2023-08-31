import tensorflow as tf
from keras import backend as K




# Load the .tflite model
model_path = 'C:/Users/MICHE/Desktop/tflite_models/optimized_model_unet.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input details
input_details = interpreter.get_input_details()
print("Input details:")
print(input_details)

# Get output details
output_details = interpreter.get_output_details()
print("Output details:")
print(output_details)

# Print other model information
print("Model input shape(s):", [detail['shape'] for detail in input_details])
print("Model output shape(s):", [detail['shape'] for detail in output_details])
print("Model input data type(s):", [detail['dtype'] for detail in input_details])
print("Model output data type(s):", [detail['dtype'] for detail in output_details])

# def mean_iou(y_true, y_pred):
#     yt0 = y_true[:,:,:,0]
#     yp0 = K.cast(y_pred[:,:,:,0] > 0.5, 'float32')
#     inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
#     union = tf.math.count_nonzero(tf.add(yt0, yp0))
#     iou = tf.where(tf.equal(union, 0), 1., tf.cast(inter/union, 'float32'))
#     return iou
# # Load your TensorFlow model
# saved_model_dir = "C:/Users/MICHE/Desktop/Master/MTP/tensorFlow/unet.h5"
# model = tf.keras.models.load_model(saved_model_dir, custom_objects={'mean_iou': mean_iou})

# # Print model summary to get an overview of the model architecture
# #model.summary()

# # Get input layer information
# input_layer = model.input
# print("Input details:")
# print("Input name:", input_layer.name)
# print("Input shape:", input_layer.shape)
# print("Input data type:", input_layer.dtype)

# # Get output layer information
# output_layer = model.output
# print("\nOutput details:")
# print("Output name:", output_layer.name)
# print("Output shape:", output_layer.shape)
# print("Output data type:", output_layer.dtype)