import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
from train_utils import (
    create_augmentor, rotate_3D, random_elastic_deformation
)

from monai.transforms import (
    RandFlipd, RandAffined, Rand3DElasticd, Rand2DElasticd, RandGaussianSmoothd,
    AddChanneld, RandGaussianNoised, Compose
)

def get_augmentations_2d():
    train_transforms = [
        AddChanneld(keys = ['img', 'seg']),
        RandFlipd(keys = ['img', 'seg'], prob = 0.1, spatial_axis = 0), 
        RandFlipd(keys = ['img', 'seg'], prob = 0.1, spatial_axis = 1),
        RandAffined(
            keys = ['img', 'seg'], mode = ('bilinear', 'nearest'),
            rotate_range = (np.pi/2, 0, 0),
            prob = 0.2, padding_mode = "zeros"),
        Rand2DElasticd(
            keys = ['img', 'seg'], mode = ('bilinear', 'nearest'),
            magnitude_range = (1, 2), #scale_range =(0.05, 0.05),
            prob = 0.2, padding_mode = 'zeros', spacing = (20, 20)),
        RandGaussianSmoothd(keys = ['img'], sigma_x = (0.5, 1.15), sigma_y = (0.5, 1.15), prob = 0.1),
        RandGaussianNoised(keys = ['img'], prob = 0.2, mean = np.random.uniform(0, 0.05), std = np.random.uniform(0, 0.1)),
    ]

    train_transforms = Compose(train_transforms)

    #test_transforms = [
    #    AddChanneld(keys = ['img', 'seg'])
    #]

    test_transforms = None

    return train_transforms, test_transforms

class Augment2D(object):
    def __init__(self):
        self.intensity_seq = create_augmentor()

    def __call__(self, sample, keys = ['img', 'seg']):
        img = sample[keys[0]]
        mask = sample[keys[1]]
        

        param = np.random.randint(0,5)
        
        if param == 0:
            # No Augmentation
            img = img
            mask = mask
            #print('No Augmentation')
        elif param == 1:
            # Apply Random Rotation
            angle = np.random.randint(-5, 5)
            img, mask = rotate_3D(img, mask, angle)
            #print('Random Rotation')
        elif param == 2:
            # Apply Random Elastic Deformation
            stacked = np.stack([img, mask], axis = 2)
            augmented = random_elastic_deformation(stacked, stacked.shape[1]*2, stacked.shape[1]*0.08)
            img = augmented[:, :, 0]
            mask = augmented[:, :, 1]
            #print('Random Elastic Deformation')
        elif param == 3:
            # Apply Intensity Adjustment
            img = self.intensity_seq.augment_image(img)
            mask = mask
            #print('Intensity Adjustment')
        else:
            # Apply Random Flip
            stacked = np.stack([img, mask], axis = 2)
            augmented = np.fliplr(stacked)
            img = augmented[:, :, 0]
            mask = augmented[:, :, 1]
            #print('Random Flip')
            
        return {'img': img, 'seg': mask}



class DataGenerator2D(Sequence):
    '''
    Issues: 
    1. Must give resize_to otherwise error
    2. Nothing implemented for expand = False
    '''
    def __init__(
        self, df, resize_to = None, transform = None, 
        expand = True, shuffle = True, batch_size = 1
    ):
        
        self.df = df
        self.expand = expand
        self.batch_size = batch_size
        self.resize_to = resize_to
        self.transform = transform
        self.shuffle = shuffle
        self.interpolation = cv2.INTER_AREA
        self.batch_shape = (batch_size, *resize_to, 1)
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))
    
    def on_epoch_end(self):
        # Update indexes after each epoch
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def get_sample(self, index):
        img_path = self.df.loc[index]['imgpath']
        mask_path = self.df.loc[index]['maskpath']
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
 
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.threshold(mask, np.min(mask), 1, cv2.THRESH_BINARY)[1]

        if self.resize_to:
            img = cv2.resize(img, self.resize_to, interpolation = self.interpolation)
            mask = cv2.resize(mask, self.resize_to, interpolation = self.interpolation)

        # Do not expand here

        sample = {'img': img, 'seg': mask}

        # Apply Transformations
        if self.transform:
            sample = self.transform(sample)

        img = sample['img']
        seg = sample['seg']

        # Normalize Pixel Values
        img = np.asarray(img, dtype=np.float32)
        img = img/np.max(img)

        return img, seg

    def get_batch(self, batch_indexes):
        X = np.empty(self.batch_shape)
        y = np.empty(self.batch_shape)

        # Move axis & Expand dims
        for i, index in enumerate(batch_indexes):
            img, seg = self.get_sample(index)
            img = np.squeeze(img)
            seg = np.squeeze(seg)
            X[i, ...] = np.expand_dims(img, -1)
            y[i, ...] = np.expand_dims(seg, -1)

        return X, y
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size: (index+1)*self.batch_size]
        X, y = self.get_batch(batch_indexes)
        return X, y
