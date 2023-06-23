from model.augmentations import RandomHorizontalFlip, RandomMirror, RandomZoom, RandomRotate, RandomShift, OverlayBox, AddNoise, GaussianBlur

class args:
    base_path = r'D:\DL-CV-ML Projects\Carvana Challenge - Instance Segmentation'
    img_size = (128, 128)
    batch_size = 32
    epochs = 3
    augment_in_ds = False # arg.add_argument("-a", "--augment", type=bool, default=False, help="Whether to augment within the dataset or use a model before the main model")
    training_method = 'train_step' # arg.add_argument("-tm", "--training_method", choices=["fit", "eagerly", "train_step"], help="Method to use to train the model")
    optimizer = 'Adam' # arg.add_argument("-o", "--optimizer")
    metrics = 'Mean' # arg.add_argument("-m", "--metrics", nargs="+")
    val_metrics = 'Mean' # arg.add_argument("-vm", "--val_metrics", nargs="+)
    monitor = 'loss_metric' # arg.add_argument

config_opt = {"learning_rate": 1e-3, "beta_1": 0.15, "beta_2": 0.99, "epsilon": 1e-8}
config_metrics = {'name': args.monitor}
config_val_metrics = {'name': "val_loss_metric"}

translate = 0.2
rot = 0.5
scale = 0.35
kernel_size = 3
sigma = 1
min_height = 50
min_width = 50
max_height = 80
max_width = 80

save_filepath = r'D:\DL-CV-ML Projects\Carvana Challenge - Instance Segmentation\Instance Segmentation\saved_model\temp_model.h5'
img_path=r'D:\DL-CV-ML Projects\Carvana Challenge - Instance Segmentation\29bb3ece3180_11.jpg'
train_tfrecord_file = r"D:\DL-CV-ML Projects\Carvana Challenge - Instance Segmentation\TFRecordFiles\train.tfrecord"

aug_functions = [
    RandomHorizontalFlip(),
    RandomShift(translate=translate),
    RandomRotate(rot=rot),
    RandomZoom(height_factor=scale, width_factor=scale),
    RandomMirror(),
    AddNoise(),
    GaussianBlur(kernel_size=kernel_size, sigma=sigma),
    OverlayBox(min_height=min_height, min_width=min_width, height=max_height, width=max_width)
]