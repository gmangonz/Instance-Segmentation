import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
# import tensorflow_addons as tfa
# from tensorflow.keras import Sequential
import matplotlib.pyplot as plt

# what I learned:
# layers.Layer sublasses work best with call() rather than __call__()

class RandomGamma(layers.Layer):

  def __init__(self, 
               gamma=0.4, 
               gain=0.75, 
               **kwargs):

    super(RandomGamma, self).__init__(**kwargs)

    self.gamma = gamma
    self.gain = gain

  def call(self, inputs):

    img, mask = inputs 

    gamma = (0, self.gamma)
    gamma = tf.random.uniform(shape=(), minval=gamma[0], maxval=gamma[1])

    gain = (self.gain/2, self.gain)
    gain = tf.random.uniform(shape=(), minval=gain[0], maxval=gain[1]) 

    img_gamma = tf.image.adjust_gamma(img, gamma=1+gamma, gain=gain)
    mask_gamma = tf.image.adjust_gamma(mask, gamma=1+gamma, gain=gain)

    return img_gamma, mask_gamma


class RandomHorizontalFlip(layers.Layer):

  def __init__(self, 
                probability,
                seed=42,
                **kwargs):

    super(RandomHorizontalFlip, self).__init__(**kwargs)
    
    self.flip_img = layers.RandomFlip(mode='vertical', seed=seed, **kwargs)
    self.flip_mask = layers.RandomFlip(mode='vertical', seed=seed, **kwargs)
    self.probability = probability

  def call(self, inputs): # img: (h, w, c)

    img, mask = inputs

    # img_flipped = self.flip_img(img)
    # mask_flipped = self.flip_mask(mask)

    # img_flipped = tf.cast(img_flipped, img.dtype)
    # mask_flipped = tf.cast(mask_flipped, mask.dtype)

    return tf.cond(tf.less(tf.random.uniform([]), self.probability), lambda: (self.flip_img(img), self.flip_mask(mask)), lambda: inputs)
  
class RandomMirror(layers.Layer):

    def __init__(self, 
                 probability,
                 **kwargs):
      
      super(RandomMirror, self).__init__(**kwargs)
      self.probability = probability
    
    def call(self, inputs):

      img, mask = inputs
    #   img_mirror = img[:, :, ::-1]
    #   mask_mirror = mask[:, :, ::-1]

    #   img_mirror = tf.cast(img_mirror, img.dtype)
    #   mask_mirror = tf.cast(mask_mirror, mask.dtype)

      return tf.cond(tf.less(tf.random.uniform([]), self.probability), lambda: (img[:, :, ::-1], mask[:, :, ::-1]), lambda: inputs) # img_mirror, mask_mirror


class RandomZoom(layers.Layer):

    def __init__(self, 
                 probability,
                 height_factor = 0.2, 
                 width_factor = 0.2, 
                 seed = 42,
                 **kwargs):

        super(RandomZoom, self).__init__(**kwargs)

        self.zoom_img = layers.RandomZoom(height_factor=height_factor, width_factor=width_factor, fill_mode='nearest', interpolation='nearest', seed=seed, fill_value=0.0)
        self.zoom_mask = layers.RandomZoom(height_factor=height_factor, width_factor=width_factor, fill_mode='nearest', interpolation='nearest', seed=seed, fill_value=0.0)
        self.probability = probability

    def call(self, inputs):

        img, mask = inputs

        # img_zoom = self.zoom_img(img)
        # mask_zoom = self.zoom_mask(mask)

        # img_zoom = tf.cast(img_zoom, img.dtype)
        # mask_zoom = tf.cast(mask_zoom, mask.dtype)

        return tf.cond(tf.less(tf.random.uniform([]), self.probability), lambda: (self.zoom_img(img), self.zoom_mask(mask)), lambda: inputs) # img_zoom, mask_zoom
    

class RandomRotate(layers.Layer):

  def __init__(self, 
               probability,
               rot=0.3,
               seed=42,
               **kwargs):

    super(RandomRotate, self).__init__(**kwargs)
    
    self.rotate_img = layers.RandomRotation(factor=rot, fill_mode='nearest', interpolation='nearest', fill_value=0.0, seed=seed)
    self.rotate_mask = layers.RandomRotation(factor=rot, fill_mode='nearest', interpolation='nearest', fill_value=0.0, seed=seed)
    self.probability = probability

  def call(self, inputs):

    img, mask = inputs
    
    # img_rotate = self.rotate_img(img)
    # mask_rotate = self.rotate_mask(mask)

    # img_rotate = tf.cast(img_rotate, img.dtype)
    # mask_rotate = tf.cast(mask_rotate, mask.dtype)

    return tf.cond(tf.less(tf.random.uniform([]), self.probability), lambda: (self.rotate_img(img), self.rotate_mask(mask)), lambda: inputs) # img_rotate, mask_rotate

class RandomShift(layers.Layer):

  def __init__(self, 
               probability,
               translate=0.2, 
               seed=42,
               **kwargs):

    super(RandomShift, self).__init__(**kwargs)

    self.translate_img = layers.RandomTranslation(translate, translate, fill_mode='nearest', interpolation='nearest', fill_value=0.0, seed=seed)
    self.translate_mask = layers.RandomTranslation(translate, translate, fill_mode='nearest', interpolation='nearest', fill_value=0.0, seed=seed)
    self.probability = probability

  def call(self, inputs): # img: (h, w, c)

    img, mask = inputs
    # img_translate = self.translate_img(img)
    # mask_translate = self.translate_mask(mask)

    # img_translate = tf.cast(img_translate, img.dtype)
    # mask_translate = tf.cast(mask_translate, mask.dtype)

    return tf.cond(tf.less(tf.random.uniform([]), self.probability), lambda: (self.translate_img(img), self.translate_mask(mask)), lambda: inputs) # img_translate, mask_translate
  

class RandomBox(layers.Layer):

    def __init__(self, 
                 min_height, 
                 min_width, 
                 max_height, 
                 max_width, 
                 **kwargs):
        
        super(RandomBox, self).__init__(**kwargs)

        self.min_height = min_height
        self.min_width = min_width
        self.height = max_height
        self.width = max_width

    def make_boxes(self, inputs): # (h, w, c), (h, w, c)

        img, masks = inputs
        img_shape = tf.shape(img)

        block_height = tf.random.uniform(shape=[], minval=self.min_height, maxval=self.height, dtype=tf.int32)
        block_width = tf.random.uniform(shape=[], minval=self.min_width, maxval=self.width, dtype=tf.int32)

        pad_h = img_shape[0] - block_height
        pad_top = tf.random.uniform(shape=[], minval=0, maxval=pad_h, dtype=tf.int32)
        pad_bottom = pad_h - pad_top

        pad_w = img_shape[1] - block_width
        pad_left = tf.random.uniform(shape=[], minval=0, maxval=pad_w, dtype=tf.int32)
        pad_right = pad_w - pad_left
        
        box_mean = tf.math.reduce_mean(img[pad_top:pad_top+block_height, pad_left:pad_left+block_width, :])
        block = tf.ones(shape=[block_height, block_width, 3]) * box_mean
        block = tf.cast(block, img.dtype)
        block_img = tf.pad(block, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=1)

        return block_img, masks

    def call(self, inputs): # (bs, h, w, c), (bs, h, w, c)
        
        imgs, masks = inputs

        block_img, masks = tf.map_fn(self.make_boxes, elems=(imgs, masks)) # Make it so that images in the batch have different boxes.   
        blocked_img = tf.multiply(tf.cast(imgs, tf.float32), tf.cast(block_img, tf.float32))
        blocked_img = tf.cast(blocked_img, imgs.dtype)
    
        return blocked_img, masks

class OverlayBox(layers.Layer):

    def __init__(self, 
                 probability, 
                 min_height, 
                 min_width, 
                 height, 
                 width, 
                 **kwargs):

        super(OverlayBox, self).__init__(**kwargs)
        
        self.probability = probability
        self.random_box = RandomBox(min_height, min_width, height, width)

    def call(self, inputs): # (bs, h, w, c), (bs, h, w, c)
        
        return tf.cond(tf.less(tf.random.uniform([]), self.probability), lambda: self.random_box(inputs), lambda: inputs)
    

class AddNoise(layers.Layer):

    def __init__(self, 
                 probability, 
                 **kwargs):

        super(AddNoise, self).__init__(**kwargs)
        self.probability = probability

    def call(self, inputs):

        img, mask = inputs

        noise = tf.random.normal(tf.shape(img), mean=0, stddev=0.075, dtype=img.dtype)
        img_noise = tf.clip_by_value(img + noise, 0.0, 1.0)
        img_noise = tf.cast(img_noise, img.dtype)

        return tf.cond(tf.less(tf.random.uniform([]), self.probability), lambda: (img_noise, mask), lambda: inputs)


class GaussianBlur(layers.Layer):

    def __init__(self, 
                 kernel_size, 
                 sigma, 
                 probability=0.75, 
                 **kwargs):
        super(GaussianBlur, self).__init__(**kwargs)

        self.kernel_size = kernel_size
        self.sigma = sigma
        self.probability = probability

    def gaussian_kernel(self, size=3, sigma=1):

        x_range = tf.range(-(size-1)//2, (size-1)//2 + 1, 1)
        y_range = tf.range((size-1)//2, -(size-1)//2 - 1, -1)

        xs, ys = tf.meshgrid(x_range, y_range)
        kernel = tf.exp(-(xs**2 + ys**2)/(2*(sigma**2))) / (2*np.pi*(sigma**2))
        return tf.cast( kernel / tf.reduce_sum(kernel), tf.float32)

    def blur_image(self, img):
       
        kernel = self.gaussian_kernel(self.kernel_size, self.sigma)
        kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)
        
        r, g, b = tf.split(img, [1,1,1], axis=-1)
        r_blur = tf.nn.conv2d(r, kernel, [1,1,1,1], 'SAME', name='r_blur')
        g_blur = tf.nn.conv2d(g, kernel, [1,1,1,1], 'SAME', name='g_blur')
        b_blur = tf.nn.conv2d(b, kernel, [1,1,1,1], 'SAME', name='b_blur')

        blur_image = tf.concat([r_blur, g_blur, b_blur], axis=-1)
        blur_image = tf.cast(blur_image, img.dtype)
        # blur_image = tfa.image.gaussian_filter2d(img, filter_shape=self.kernel_size, sigma=self.sigma, padding='CONSTANT')

        return blur_image

    def call(self, inputs):

        img, mask = inputs
        return tf.cond(tf.less(tf.random.uniform([]), self.probability), lambda: (self.blur_image(img), mask), lambda: inputs) # blur_image, mask
    

# def step(values): # "hard sigmoid", useful for binary accuracy calculation from logits. negative values -> 0.0, positive values -> 1.0  
#     return 0.5 * (1.0 + tf.sign(values))

class Ada(tf.keras.Model): # layers.Layer

    def __init__(self, 
                 translate=0.2, 
                 rot=0.5, 
                 scale=0.25, 
                 p=0.5,
                 img_size=(128, 128),
                 batch_p=0.0,
                 kernel_size=3,
                 sigma=1,
                 min_height=50, 
                 min_width=50, 
                 max_height=80, 
                 max_width=80,
                 **kwargs):
        
        super(Ada, self).__init__(**kwargs)
        
        self.probability = tf.Variable(batch_p, trainable=False) # True
        self.target_accuracy = 0.85 # 0.85, 0.95
        self.integration_steps = 500 # 1000, 1500, 2000
    
        self.augmenter = custom_data_augmentation_func(translate=translate, rot=rot, scale=scale, p=p, img_size=img_size, kernel_size=kernel_size, 
                                                       sigma=sigma, min_height=min_height, min_width=min_width, max_height=max_height, max_width=max_width)
    
    def call(self, inputs, training=False):

        # imgs, masks = inputs # [bs, H, W, C], [bs, H, W, C]

        if training:
          
          batch_size = tf.shape(imgs)[0]
          augmented_imgs, augmented_masks = self.augmenter(inputs, training) # (2, bs, H, W, C)
          augmentation_values = tf.random.uniform(shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0) # Generate random numbers
          augmentation_bools = tf.math.less(augmentation_values, self.probability) # Get booleans in the indices where we want augmented images or not
          imgs = tf.where(augmentation_bools, augmented_imgs, imgs)
          masks = tf.where(augmentation_bools, augmented_masks, masks)

        return inputs

    def update(self, loss):

        # The more accurate the model, the more augmentations is performed.
        accuracy_error = (1. - loss) - self.target_accuracy
        self.probability.assign( tf.clip_by_value(self.probability + accuracy_error / self.integration_steps, 0.0, 1.0) )



def custom_data_augmentation_func(translate = 0.2,
                                  rot = 0.5,
                                  scale = 0.25,
                                  p = 0.5,
                                  img_size = (128, 128),
                                  kernel_size = 3,
                                  sigma = 1,
                                  min_height = 50,
                                  min_width = 50,
                                  max_height = 80,
                                  max_width = 80):

    if len(img_size) == 2:
        img_shape = img_size + (3,)
        mask_shape = img_size + (1,)
    
    if len(img_size) == 3:
        img_shape = img_size
        mask_shape = (img_size[0], img_size[1], 1)

    input_img = layers.Input(shape=img_shape) # (H, W, C)
    input_mask = layers.Input(shape=mask_shape) # (H, W, C)
    x = (input_img, input_mask)
    x = RandomHorizontalFlip(probability=p)(x)
    x = RandomShift(translate=translate, probability=p)(x)
    x = RandomRotate(probability=p, rot=rot)(x)
    x = RandomZoom(probability=p, height_factor=scale, width_factor=scale)(x)
    x = RandomMirror(probability=p)(x)
    x = AddNoise(probability=p)(x)
    x = GaussianBlur(kernel_size=kernel_size, sigma=sigma, probability=p)(x)
    x = OverlayBox(probability=p, min_height=min_height, min_width=min_width, height=max_height, width=max_width)(x)
    augmenter = tf.keras.Model([input_img, input_mask], x, name='custom_data_augmentation_function')

    return augmenter







# def test_augmentation(input_test, idx=83):


#     augmenter = custom_data_augmentation_func(translate = 0.2, rot = 0.5, scale = 0.2, p=0.5)
#     a, b = augmenter(input_test)
#     plt.imshow(a[idx])
#     plt.show()
#     plt.imshow(b[idx])
#     plt.show()

#     return a, b


        # if len(img_size) == 2:
        #    img_shape = img_size + (3,)
        #    mask_shape = img_size + (1,)
        
        # if len(img_size) == 3:
        #     img_shape = img_size
        #     mask_shape = (img_size[0], img_size[1], 1)
            
        # input_img = layers.Input(shape=img_shape) # dtype=tf.float32 (H, W, C)
        # input_mask = layers.Input(shape=mask_shape) # (H, W, C)
        # x = (input_img, input_mask)
        # x = RandomHorizontalFlip(probability=p)(x)
        # x = RandomShift(translate=translate, probability=p)(x)
        # x = RandomRotate(probability=p, rot=rot)(x)
        # x = RandomZoom(probability=p, height_factor=scale, width_factor=scale)(x)
        # x = RandomMirror(probability=p)(x)
        # x = AddNoise(probability=p)(x)
        # x = GaussianBlur(kernel_size=kernel_size, sigma=sigma, probability=p)(x)
        # x = OverlayBox(probability=p, min_height=min_height, min_width=min_width, height=max_height, width=max_width)(x)
        # self.augmenter = tf.keras.Model([input_img, input_mask], x)