from tensorflow.keras.utils import plot_model
import tensorflow as tf

def fire_module(x, squeeze_filters, expand_filters):
    squeeze = tf.keras.layers.Conv2D(squeeze_filters, (1, 1), activation='relu')(x)
    expand_1x1 = tf.keras.layers.Conv2D(expand_filters, (1, 1), activation='relu')(squeeze)
    expand_3x3 = tf.keras.layers.Conv2D(expand_filters, (3, 3), padding='same', activation='relu')(squeeze)
    return tf.keras.layers.Concatenate()([expand_1x1, expand_3x3])

input_shape = (224, 224, 3)
num_classes = 10

inputs = tf.keras.Input(shape=input_shape)

x = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='valid')(inputs)
x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

x = fire_module(x, 16, 64)
x = fire_module(x, 16, 64)
x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

x = fire_module(x, 32, 128)
x = fire_module(x, 32, 128)
x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

x = fire_module(x, 48, 192)
x = fire_module(x, 48, 192)
x = fire_module(x, 64, 256)
x = fire_module(x, 64, 256)

x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='relu')(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Softmax()(x)

model = tf.keras.Model(inputs, outputs)
plot_model(model, to_file='squeezenet.png', show_shapes=True, show_layer_names=True)

model.summary()


