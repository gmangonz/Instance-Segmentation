import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class DisplayCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, 
                 img_path,
                 args):

        super(DisplayCallback, self).__init__()
        
        self.test_img_for_epoch_viz = img_to_array(load_img(img_path, target_size=args.img_size)) / 255
        self.training_method = args.training_method
    
    def on_epoch_begin(self, epoch, logs=None):
        
        if (epoch + 1) % 3 == 0 or epoch == 0:
            
            if self.training_method == 'train_step': 
                test_pred_mask = self.model.model(self.test_img_for_epoch_viz[None, ...], training=False)
            
            if self.training_method == 'fit':
                test_pred_mask = self.model(self.test_img_for_epoch_viz[None, ...], training=False)
            
            plt.figure(figsize=(8, 8))
            title = ['Input Image', 'Predicted Mask']
            display_list = [self.test_img_for_epoch_viz, test_pred_mask[0]]
            for i in range(len(display_list)):
                plt.subplot(1, len(display_list), i+1)
                plt.title(title[i])
                plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
                plt.axis('off')
            plt.show()