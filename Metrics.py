from tensorflow.keras import backend as K
import tensorflow as tf

def dice_coef(y_true, y_pred, smooth=100):

    y_true_f = K.flatten(y_true/255)
    y_pred_f = K.flatten(y_pred/255)

    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return dice

def dice_loss(y_true, y_pred):
    '''
    Loss function
    '''
    y_pred = tf.cast(y_pred, y_true.dtype) # Might be able to remove
    loss = 1 - dice_coef(y_true, y_pred)
    return loss