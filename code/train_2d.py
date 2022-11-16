# -------------------------- Library Imports -----------------------------------

import wandb
from wandb.keras import WandbCallback
import pandas as pd
from tqdm import tqdm
from train_utils import add_backup_callback, create_list_of_data_2d
from data_utils import DataGenerator2D, Augment2D, get_augmentations_2d
from model_lib import UNet2D_4x, dice_coef
from tensorflow.keras.callbacks import (
    CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
from tensorflow.keras.optimizers import Adam


# -------------------------- Initial Configurations ----------------------------
trainpath = "C:/Data/VIP2018/train/"
valpath = "C:/Data/VIP2018/valid/"

img_rows = 256
img_cols = 256

smooth = 1.
batch_size = 32

epochs = 50

lr = 1e-3
optim = 'Adam'

checkpoint_dir = '/content/drive/MyDrive/DataBuffer/checkpoints/'
checkpoint_dir = 'C:\\Data\\models\\'

model_name = 'UNet2D_4x'

wandb_log = False

# -------------------------- Wandb Setup ---------------------------------

id = wandb.util.generate_id()
#id = ''

print('Please store the following run id for resuming in the future ..')
print(id)

project_name = "NSCLC-Old-Pipe-3D"
run_name = "R3DDUNet_Spectrum"

transform_str = ""

wandb_config = {
    'learning_rate': lr,
    'optimizer': optim,
    'batch_size': batch_size,
    'epochs': epochs,
    'resolution': img_rows,
    'dataset': 'Spectrum 2D with Aug',
    'augmentation': transform_str,
}

if(wandb_log):
    run = wandb.init(
        project = project_name,
        config = wandb_config,
        resume = "allow", id = id,
        name = run_name,
    )

# -------------------------- Data Preparation -----------------------------------

train_image_paths, train_mask_paths = create_list_of_data_2d(trainpath, shuffle = False)
val_image_paths, val_mask_paths = create_list_of_data_2d(valpath, shuffle = False)

train_df = pd.DataFrame(
    {'imgpath': train_image_paths, 'maskpath': train_mask_paths}
)

val_df = pd.DataFrame(
    {'imgpath': val_image_paths, 'maskpath': val_mask_paths}
)

train_df.to_pickle('train_2d.pkl')
val_df.to_pickle('val_2d.pkl')

# train_df = pd.read_pickle('train_2d.pkl')
# val_df = pd.read_pickle('val_2d.pkl')

train_df = train_df.sample(frac=1).reset_index(drop=True)
val_df = val_df.sample(frac=1).reset_index(drop=True)

# Lets use Monai transformations instead of imgaug
train_transforms, test_transforms = get_augmentations_2d()


train_gen = DataGenerator2D(
    train_df, resize_to = (img_rows, img_cols),
    batch_size = batch_size, transform = train_transforms,
)

val_gen = DataGenerator2D(
    val_df, resize_to = (img_rows, img_cols),
    batch_size = batch_size, transform = test_transforms,
)

# -------------------------- Model Setup -----------------------------------

model = UNet2D_4x()

# -------------------------- Callbacks ---------------------------------

es_cb = EarlyStopping(
    monitor='val_dice_coef', patience=15, verbose=1,
    min_delta=1e-6, mode = 'max')
lr_cb = ReduceLROnPlateau(
    monitor='val_dice_coef', factor=0.5,patience=3, verbose=1,
    mode='max', cooldown=0, min_lr=1e-8)
ckp_cb = ModelCheckpoint(
    checkpoint_dir + model_name, monitor='val_dice_coef', verbose=1,
    save_weights_only = False, save_best_only=True, mode='max', save_format = 'tf')
csv_cb = CSVLogger(
    checkpoint_dir + 'training_' + model_name + '.log')

callbacks = {
    'model_checkpoint': ckp_cb,
    'EarlyStopping': es_cb,
    'ReduceLROnPlateau': lr_cb,
    'CSVLogger': csv_cb
}

callbacks = add_backup_callback(callbacks, checkpoint_dir + '/backup')

if wandb_log:
    callbacks.append(WandbCallback(save_model = False))

# -------------------------- Model Training -----------------------------------

model.compile(
    optimizer=Adam(
        lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199),
    loss = 'binary_crossentropy',
    metrics=[dice_coef]
)

model.fit( 
    train_gen, 
    steps_per_epoch  = int(len(train_df)/batch_size), # I made a change here
    epochs           = epochs, 
    verbose          = 1,
    validation_data  = val_gen,
    validation_steps = int(len(val_df)/batch_size),
    callbacks        = callbacks, 
    workers          = 0,
    max_queue_size   = 8)

# -------------------------- Cleanup ---------------------------------

if(wandb_log):
    run.finish()