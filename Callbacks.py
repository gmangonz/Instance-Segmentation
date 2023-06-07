import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings

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
            
            if self.training_method in ['fit', 'eagerly']:
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


class SaveModel(tf.keras.callbacks.ModelCheckpoint):

    def __init__(self, 
                 model_to_save, 
                 **kwargs):
        
        super(SaveModel, self).__init__(**kwargs)
        self.model_to_save = model_to_save

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        filepath = self.filepath.format(epoch=epoch + 1, **logs)
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can save best model only with %s available, '
                                'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print(f'\nEpoch {epoch+1}: {self.monitor} improved from {self.best} to {current}, saving model to {filepath}')
                    self.best = current
                    if self.save_weights_only:
                        self.model_to_save.save_weights(filepath, overwrite=True)
                    else:
                        self.model_to_save.save(filepath, overwrite=True)
                else:
                    if self.verbose > 0:
                        print(f'\nEpoch {epoch+1}: {self.monitor} did not improve from {self.best}')
        else:
            if self.verbose > 0:
                print(f'\nEpoch {epoch+1}: saving model to {filepath}')
            if self.save_weights_only:
                self.model_to_save.save_weights(filepath, overwrite=True)
            else:
                self.model_to_save.save(filepath, overwrite=True)

        super(SaveModel, self).on_batch_end(epoch, logs)


def create_callbacks(filepath_name, args, model_to_save):
    
    early_stop = EarlyStopping(
        monitor              = args.monitor, 
        min_delta            = 0.01, 
        patience             = 5, 
        mode                 = 'min', 
        restore_best_weights = True)

    checkpoint = SaveModel(
        model_to_save      = model_to_save,
        filepath           = filepath_name,
        monitor            = args.monitor, 
        verbose            = 1, 
        save_best_only     = True, 
        save_weights_only  = True,
        mode               = 'min', 
        save_freq          = 'epoch'
    )

    reduce_on_plateau = ReduceLROnPlateau(
        monitor   = args.monitor,
        factor    = 0.1,
        patience  = 5,
        verbose   = 1,
        mode      = 'min',
        min_delta = 0.01,
        cooldown  = 0,
        min_lr    = 0
    )

    display = DisplayCallback(
        img_path = r'D:\DL-CV-ML Projects\Carvana Challenge - Instance Segmentation\29bb3ece3180_11.jpg', 
        args     = args
        )
    return [early_stop, checkpoint, reduce_on_plateau, display]