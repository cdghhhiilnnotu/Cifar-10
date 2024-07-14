# class CifarModel(Model):
#     def __init__(self):
#         super(CifarModel, self).__init__()
#         self.conv2d_1 = keras.layers.Conv2D(32,(3,3),activation='relu')
#         self.batchnor_1 = keras.layers.BatchNormalization()
#         self.mpool_1 = keras.layers.MaxPooling2D(pool_size=(2,2))

#         self.conv2d_2 = keras.layers.Conv2D(64,(3,3),activation='relu')
#         self.batchnor_2 = keras.layers.BatchNormalization()
#         self.mpool_2 = keras.layers.MaxPooling2D(pool_size=(2,2))

#         self.conv2d_3 = keras.layers.Conv2D(32,(3,3),activation='relu')
#         self.batchnor_3 = keras.layers.BatchNormalization()
#         self.mpool_3 = keras.layers.MaxPooling2D(pool_size=(2,2))

#         self.flatten = keras.layers.Flatten()
#         self.dense_1 = keras.layers.Dense(16,activation='relu')

#         self.output = keras.layers.Dense(10,activation='softmax')

#         def __call__(self, input):
#         # Define the forward pass
#             x = self.conv2d_1(input)
#             x = self.batchnor_1(x)
#             x = self.mpool_1(x)
#             x = self.conv2d_2(x)
#             x = self.batchnor_2(x)
#             x = self.mpool_2(x)
#             x = self.conv2d_3(x)
#             x = self.batchnor_3(x)
#             x = self.mpool_3(x)
#             x = self.flatten(x)
#             x = self.dense_1(x)

#             return self.output(x)
        
# model = CifarModel()
    
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

import tensorflow as tf
from cifar_lib_0 import *
from keras import layers

# Define the custom model class
class CifarModel(Model):
    def __init__(self):
        super(CifarModel, self).__init__()
        self.conv2d_1 = keras.layers.Conv2D(32,(3,3),activation='relu')
        self.batchnor_1 = keras.layers.BatchNormalization()
        self.mpool_1 = keras.layers.MaxPooling2D(pool_size=(2,2))

        self.conv2d_2 = keras.layers.Conv2D(64,(3,3),activation='relu')
        self.batchnor_2 = keras.layers.BatchNormalization()
        self.mpool_2 = keras.layers.MaxPooling2D(pool_size=(2,2))

        self.conv2d_3 = keras.layers.Conv2D(32,(3,3),activation='relu')
        self.batchnor_3 = keras.layers.BatchNormalization()
        self.mpool_3 = keras.layers.MaxPooling2D(pool_size=(2,2))

        self.flatten = keras.layers.Flatten()
        self.dense_1 = keras.layers.Dense(16,activation='relu')

        self.output_layer = keras.layers.Dense(10,activation='softmax')

    def __call__(self, input):
        # Define the forward pass
        x = self.conv2d_1(input)
        x = self.batchnor_1(x)
        x = self.mpool_1(x)
        x = self.conv2d_2(x)
        x = self.batchnor_2(x)
        x = self.mpool_2(x)
        x = self.conv2d_3(x)
        x = self.batchnor_3(x)
        x = self.mpool_3(x)
        x = self.flatten(x)
        x = self.dense_1(x)

        return self.output_layer(x)

# Instantiate the model
model = CifarModel()

# # Define input shape
# input_shape = (None, 32, 32, 3)

# # Build the model by calling it with a sample input
# model.build(input_shape)

# # Alternatively, you can call the model with a sample input
# sample_input = tf.random.normal([1, 32, 32, 3])
# model(sample_input)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# # Example data
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# x_train = x_train.astype("float32") / 255
# x_test = x_test.astype("float32") / 255

# # Train the model
# model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

