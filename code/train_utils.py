import os
import json
import logging

import cv2
import random

import numpy as np

from tqdm import tqdm


logger = logging.getLogger(__name__)


class CustomBackupAndRestore(tf.keras.callbacks.experimental.BackupAndRestore):
    """This callback is experimental and might be removed in future.
    See :any:`add_backup_callback`
    """

    def __init__(self, callbacks, backup_dir, **kwargs):
        super().__init__(backup_dir=backup_dir, **kwargs)
        self.callbacks = callbacks
        self.callbacks_backup_path = os.path.join(self.backup_dir, "callbacks.json")

    def backup(self):
        variables = {}
        for cb_name, cb in self.callbacks.items():
            variables[cb_name] = {}
            for k, v in cb.__dict__.items():
                if not isinstance(v, (int, float)):
                    continue
                variables[cb_name][k] = v
        with open(self.callbacks_backup_path, "w") as f:
            json.dump(variables, f, indent=4, sort_keys=True)

    def restore(self):
        if not os.path.isfile(self.callbacks_backup_path):
            return False

        with open(self.callbacks_backup_path, "r") as f:
            variables = json.load(f)

        for cb_name, cb in self.callbacks.items():
            if cb_name not in variables:
                continue
            for k, v in cb.__dict__.items():
                if k in variables[cb_name]:
                    cb.__dict__[k] = variables[cb_name][k]

        return True

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs=logs)
        if self.restore():
            logger.info(f"Restored callbacks from {self.callbacks_backup_path}")
        else:
            logger.info("Did not restore callbacks")

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs=logs)
        self.backup()

    def on_train_end(self, logs=None):
        # do not delete backups
        pass



def add_backup_callback(callbacks, backup_dir, **kwargs):
    """Adds a backup callback to your callbacks to restore the training process
    if it is interrupted.

    .. warning::

        This function is experimental and may be removed or changed in future.

    Examples
    --------

    >>> CHECKPOINT = "checkpoints"
    >>> callbacks = {
    ...     "best": tf.keras.callbacks.ModelCheckpoint(
    ...         f"{CHECKPOINT}/best",
    ...         monitor="val_acc",
    ...         save_best_only=True,
    ...         mode="max",
    ...         verbose=1,
    ...     ),
    ...     "tensorboard": tf.keras.callbacks.TensorBoard(
    ...         log_dir=f"{CHECKPOINT}/logs",
    ...         update_freq=15,
    ...         write_graph=False,
    ...     ),
    ... }
    >>> callbacks = add_backup_callback(callbacks, f"{CHECKPOINT}/backup")
    >>> # callbacks will be a list that can be given to model.fit
    >>> isinstance(callbacks, list)
    True
    """
    if not isinstance(callbacks, dict):
        raise ValueError(
            "Please provide a dictionary of callbacks where "
            "keys are simple names for your callbacks!"
        )
    cb = CustomBackupAndRestore(callbacks=callbacks, backup_dir=backup_dir, **kwargs)
    callbacks = list(callbacks.values())
    callbacks.append(cb)
    return callbacks

def create_list_of_data_2d(data_path, shuffle = False, seed = 42):
    image_paths = []
    mask_paths = []

    patient_list = os.listdir(data_path)

    for patient in tqdm(patient_list):
        patient_path = os.path.join(data_path, patient)
        images_path = os.path.join(patient_path, 'images')
        masks_path = os.path.join(data_path, patient, 'masks')
        masks = sorted(os.listdir(masks_path))
        
        for mask in masks:
            mask_path = os.path.join(masks_path, mask)
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if np.min(mask_img) != np.max(mask_img):
                img_path = os.path.join(images_path, mask)
                image_paths.append(img_path)
                mask_paths.append(mask_path)
        
    if shuffle:
        all_paths = list(zip(image_paths, mask_paths))
        random.seed(seed)
        random.shuffle(all_paths)
        image_paths, mask_paths = zip(*all_paths)
    
    return image_paths, mask_paths
