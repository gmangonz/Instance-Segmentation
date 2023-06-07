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


class MyModel(tf.keras.Model):

    def __init__(self, 
                 img_size,
                 augment_func, # ada_p=0.5, ada_batch_p=0.0,
                 **kwargs):
               
        super(MyModel, self).__init__(**kwargs)

        self.img_size = img_size
        self.train_step_counter = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.model = make_dummy_model(img_size)
        self.ada = augment_func
 
    def compile(self, optimizer=None, metrics=[], *args, **kwargs):

        assert isinstance(metrics, list), "metrics input must be a list"
        self.train_step_counter.assign(0)
        self.optimizer = optimizer
        self.loss_tracker = metrics[0] if len(metrics) == 1 else tf.keras.metrics.Mean(name="loss_metric") ###### Make it so metrics can be multiple items ######
        self.dice_coeff_tracker = tf.keras.metrics.Mean(name="dice_coeff_metric")     
        self.augmentation_probability_tracker = tf.keras.metrics.Mean(name="aug_probability")

        super(MyModel, self).compile(*args, **kwargs)

    @property
    def metrics(self):

        return [self.loss_tracker, self.dice_coeff_tracker, self.augmentation_probability_tracker]
    
    def train_step(self, ds_input): # If you pass a tf.data.Dataset, by calling fit(dataset, ...), then data will be what gets yielded by dataset at each batch.

        self.train_step_counter.assign_add(1)
        augmented_images, augmented_masks = self.ada(ds_input, training=True) # out: tf.float32, tf.int32 <- in: tf.float32, tf.int32
        
        with tf.GradientTape() as tape: # WARNING:tensorflow:The dtype of the target tensor must be floating (e.g. tf.float32) when calling GradientTape.gradient, got tf.int32

            predicted = self.model(augmented_images, training=True) # out: tf.float32 <- in: tf.float32
            loss = self.compiled_loss(augmented_masks, predicted) # out: tf.float32 <- in: tf.float32, tf.float32 IF loss is tf.int32 THIS WILL GIVE ERROR OF: No gradients provided for any variable, LOSS FUNCTION NEED TO RETURN FLOAT32, I THINK
            
        trainable_weights = self.model.trainable_variables # self.model.trainable_weights
        model_grads = tape.gradient(loss, trainable_weights)
        self.optimizer.apply_gradients(zip(model_grads, trainable_weights))
        
        self.ada.update(loss)
        self.loss_tracker.update_state(loss)
        self.dice_coeff_tracker.update_state(1-loss)     
        self.augmentation_probability_tracker.update_state(self.ada.probability)
        
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training=False, augment=False):

        augmented_images, augmented_masks = self.ada(inputs, training=training) if augment else inputs
        predicted = self.model(augmented_images, training=training)
        return augmented_images, augmented_masks, predicted