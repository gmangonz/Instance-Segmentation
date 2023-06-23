from model.Model import make_dummy_model, MyModel
from model.augmentations import Ada, build_augmenter
from tools.data_prep import get_data_df, data_generator, get_train_val, data_generator_tfrecordFile
import tensorflow as tf
import warnings

def get_data(args, from_tfRecords, train_tfrecord_file = None, val_tfrecord_file = None, aug_functions=None):


    if args.augment_in_ds == True: 
        assert aug_functions is not None, "a list of augmentation functions should be provided if you want to augment in the dataset"
        ds_augment_func = build_augmenter(aug_functions, args.img_size)

    if args.augment_in_ds == False: 
        ds_augment_func = None

    df_data, df_train_mask = get_data_df(args)
    df_train, df_val = get_train_val(df_data, aug=True)

    train_img_list = df_train['path'].to_list()
    train_mask_list = df_train['mask_path'].to_list()

    val_img_list = df_val['path'].to_list()
    val_mask_list = df_val['mask_path'].to_list()

    if from_tfRecords: 
        assert (train_tfrecord_file != None), 'to load from tfRecord file, train_tfrecord_file and val_tfrecord_file must be specified'

        ds_train = data_generator_tfrecordFile(train_tfrecord_file, split='train', img_size=args.img_size, ds_augment_func=ds_augment_func, batch_size=args.batch_size)
        ds_val = data_generator(image_list=val_img_list, mask_list=val_mask_list, split='val', img_size=args.img_size, ds_augment_func=ds_augment_func, batch_size=args.batch_size)
    
        return ds_train, ds_val

    ds_train = data_generator(image_list=train_img_list, mask_list=train_mask_list, split='train', img_size=args.img_size, ds_augment_func=ds_augment_func, batch_size=args.batch_size)
    ds_val = data_generator(image_list=val_img_list, mask_list=val_mask_list, split='val', img_size=args.img_size, ds_augment_func=ds_augment_func, batch_size=args.batch_size)

    return ds_train, ds_val


def setup_model(args, aug_functions):

    assert isinstance(aug_functions, list) 

    switch = False if args.augment_in_ds else True
    augment_func = Ada(img_size=args.img_size, aug_functions=aug_functions, initial_probability=0.0, switch=switch)


    if args.training_method == 'train_step': 
        
        my_model = MyModel(img_size=args.img_size, augment_func=augment_func, args=args)
        return None, my_model

    else:

        my_model = make_dummy_model(img_size=args.img_size)
        if args.training_method == 'fit' and args.augment_in_ds == False: ##### ADD ARGUMENT THAT SPECIFIES IF WE WANT DATA AUGMENTATION #####
            warnings.warn('Cannot have args.augment set to True, args.augment_in_ds set to False with args.training_method set to fit, will not perform data augmentation')

        return augment_func, my_model

def set_up_to_train(args, config_opt, config_metrics, config_val_metrics):

    # Keep track of metrics with Mean
    augmentation_probability_tracker = tf.keras.metrics.Mean(name="aug_probability")
    dice_coeff_tracker = tf.keras.metrics.Mean(name="dice_coeff_metric")
    val_dice_coeff_tracker = tf.keras.metrics.Mean(name="val_dice_coeff_metric")

    # Get optimizer
    optimizer = tf.keras.optimizers.get(args.optimizer)
    optimizer = optimizer.from_config(config=config_opt)

    # Get loss tracker
    loss_tracker = tf.keras.metrics.get(args.metrics)
    loss_tracker = loss_tracker.from_config(config=config_metrics)

    # Get val loss tracker
    val_loss_tracker = tf.keras.metrics.get(args.val_metrics)
    val_loss_tracker = val_loss_tracker.from_config(config=config_val_metrics)

    return optimizer, augmentation_probability_tracker, dice_coeff_tracker, loss_tracker, val_dice_coeff_tracker, val_loss_tracker