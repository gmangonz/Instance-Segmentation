import tensorflow as tf
from tensorflow.keras import layers

def make_dummy_model(img_size):

    if len(img_size) == 2:
        shape = (img_size[0], img_size[1], 3)
    if len(img_size) == 3:
        shape = img_size

    inputs = tf.keras.Input(shape=shape)
    x = layers.Conv2D(32, kernel_size=3, strides=1, padding="same", name='Conv1')(inputs)
    x = layers.BatchNormalization(name='BN')(x)
    x = layers.Activation("relu", name='Act')(x)
    outputs = layers.Conv2D(filters=1, kernel_size=3, activation="sigmoid", padding="same", name='Conv2')(x)

    model = tf.keras.Model(inputs, outputs)
    return model


# class Make_Dummy_Model(layers.Layer):

#     def __init__(self, img_size, **kwargs):
        
#         super(Make_Dummy_Model, self).__init__(**kwargs)
#         self.dummy_model = make_dummy_model(img_size)

#         # self.inputs = tf.keras.Input(shape=shape)
#         self.layer = layers.Conv2D(32, kernel_size=3, strides=1, padding="same")
#         self.bn = layers.BatchNormalization()
#         self.act = layers.Activation("relu")
#         self.outputs = layers.Conv2D(filters=1, kernel_size=3, activation="sigmoid", padding="same")

#     def call(self, imgs):

#         x = self.layer(imgs)
#         x = self.bn(x)
#         x = self.act(x)
#         x = self.outputs(x)

#         return x