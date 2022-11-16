from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input, concatenate, Conv3D, Conv2D, Conv2DTranspose, SpatialDropout3D,
    ConvLSTM2D, TimeDistributed, BatchNormalization, Activation,
    Permute, MaxPooling2D, SpatialDropout2D, Layer, add)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


## Configs

img_rows = 256
img_cols = 256
depth = 8
smooth = 1.

#loss function
def dice_coef(y_true, y_pred):
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

#model definitions

def dense_conv_block_2d(input_tensor, n_filters, name = None):
    x = Conv2D(n_filters, (3,3), padding = 'same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = concatenate([input_tensor, x], axis = 3)
    x = Conv2D(n_filters, (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = concatenate([input_tensor, x], axis = 3, name = name)
    return x

def get_dense_unet2D_3x(input_shape = (256, 256, 1)):
    depth_cnn = [32, 64, 128, 256]

    input = Input(input_shape)
    x1 = dense_conv_block_2d(input, depth_cnn[0], name = 'x1')
    x1_down = MaxPooling2D((2,2), name = 'x1_down')(x1)
    x1_down = SpatialDropout2D(0.1)(x1_down)

    x2 = dense_conv_block_2d(x1_down, depth_cnn[1], name = 'x2')
    x2_down = MaxPooling2D((2,2), name = 'x2_down')(x2)
    x2_down = SpatialDropout2D(0.1)(x2_down)

    x3 = dense_conv_block_2d(x2_down, depth_cnn[2], name = 'x3')
    x3_down = MaxPooling2D((2,2), name = 'x3_down')(x3)
    x3_down = SpatialDropout2D(0.1)(x3_down)

    base = dense_conv_block_2d(x3_down, depth_cnn[3], name = 'base')

    base_up = Conv2DTranspose(depth_cnn[2], (2,2), strides = (2,2), padding = 'same', name = 'base_up')(base)
    c3 = concatenate([base_up, x3], axis = 3, name = 'base_up__x3')
    d3 = dense_conv_block_2d(c3, depth_cnn[2], name = 'd3')

    d3_up = Conv2DTranspose(depth_cnn[1], (2,2), strides = (2,2), padding = 'same', name = 'd3_up')(d3)
    c2 = concatenate([d3_up, x2], axis = 3, name = 'd3_up__x2')
    d2 = dense_conv_block_2d(c2, depth_cnn[1], name = 'd2')

    d2_up = Conv2DTranspose(depth_cnn[0], (2,2), strides = (2,2), padding = 'same', name = 'd2_up')(d2)
    c1 = concatenate([d2_up, x1], axis = 3, name = 'd2_up__x1')
    d1 = dense_conv_block_2d(c1, depth_cnn[0], name = 'd1')

    out = Conv2D(1, (1,1), activation = 'sigmoid', name = 'out')(d1)

    model = Model(inputs = [input], outputs = [out])
    return model

def get_3D_Recurrent_DenseUnet():
    
    inputs = Input((img_rows, img_cols, depth, 1))
    
    #list of number of filters per block
    depth_cnn = [32, 64, 128, 256]
    
    ##start of encoder block
    
    ##encoder block1
    conv11 = Conv3D(depth_cnn[0], (3, 3, 3), padding='same', name = 'conv1_1')(inputs)
    conv11 = BatchNormalization()(conv11)
    conv11 = Activation('relu')(conv11)
    conc11 = concatenate([inputs, conv11], axis=4)
    conv12 = Conv3D(depth_cnn[0], (3, 3, 3), padding='same', name = 'conv1_2')(conc11)
    conv12 = BatchNormalization()(conv12)
    conv12 = Activation('relu')(conv12)
    conc12 = concatenate([inputs, conv12], axis=4)
    perm = Permute((3,1,2,4))(conc12)
    pool1 = TimeDistributed(MaxPooling2D((2, 2)), name = 'pool1')(perm)
    pool1 = Permute((2,3,1,4))(pool1)

    pool1 = SpatialDropout3D(0.1)(pool1)

    #encoder block2
    conv21 = Conv3D(depth_cnn[1], (3, 3, 3), padding='same', name = 'conv2_1')(pool1)
    conv21 = BatchNormalization()(conv21)
    conv21 = Activation('relu')(conv21)
    conc21 = concatenate([pool1, conv21], axis=4)
    conv22 = Conv3D(depth_cnn[1], (3, 3, 3), padding='same', name = 'conv2_2')(conc21)
    conv22 = BatchNormalization()(conv22)
    conv22 = Activation('relu')(conv22)
    conc22 = concatenate([pool1, conv22], axis=4)   
    perm = Permute((3,1,2,4))(conc22)
    pool2 = TimeDistributed(MaxPooling2D((2, 2)), name = 'pool2')(perm)
    pool2 = Permute((2,3,1,4))(pool2)

    pool2 = SpatialDropout3D(0.1)(pool2)

    #encoder block3
    conv31 = Conv3D(depth_cnn[2], (3, 3, 3), padding='same', name = 'conv3_1')(pool2)
    conv31 = BatchNormalization()(conv31)
    conv31 = Activation('relu')(conv31)
    conc31 = concatenate([pool2, conv31], axis=4)
    conv32 = Conv3D(depth_cnn[2], (3, 3, 3), padding='same', name = 'conv3_2')(conc31)
    conv32 = BatchNormalization()(conv32)
    conv32 = Activation('relu')(conv32)
    conc32 = concatenate([pool2, conv32], axis=4)  
    perm = Permute((3,1,2,4))(conc32)
    pool3 = TimeDistributed(MaxPooling2D((2, 2)), name = 'pool3')(perm)

    pool3 = SpatialDropout3D(0.1)(pool3)
    
    ##end of encoder block
    
    #ConvLSTM block 
    x = BatchNormalization()(ConvLSTM2D(filters =depth_cnn[3], kernel_size = (3,3), padding='same', return_sequences=True)(pool3))
    x = BatchNormalization()(ConvLSTM2D(filters =depth_cnn[3], kernel_size = (3,3), padding='same', return_sequences=True)(x))
    x = BatchNormalization()(ConvLSTM2D(filters = depth_cnn[3], kernel_size = (3,3), padding='same', return_sequences=True)(x))


    # start of decoder block
    
    # decoder block1
    up1 = TimeDistributed(Conv2DTranspose(depth_cnn[2], (2, 2), strides=(2, 2), padding='same', name = 'up1'))(x)   
    up1 = Permute((2,3,1,4))(up1)
    up6 = concatenate([up1, conc32], axis=4)
    conv61 = Conv3D(depth_cnn[2], (3, 3, 3), padding='same', name = 'conv4_1')(up6)
    conv61 = BatchNormalization()(conv61)
    conv61 = Activation('relu')(conv61)
    conc61 = concatenate([up6, conv61], axis=4)
    conv62 = Conv3D(depth_cnn[2], (3, 3, 3), padding='same', name = 'conv4_2')(conc61)
    conv62 = BatchNormalization()(conv62)
    conv62 = Activation('relu')(conv62)
    conv62 = concatenate([up6, conv62], axis=4)

    #decoder block2
    up2 = Permute((3,1,2,4))(conv62)
    up2 = TimeDistributed(Conv2DTranspose(depth_cnn[1], (2, 2), strides=(2, 2), padding='same'), name = 'up2')(up2)
    up2 = Permute((2,3,1,4))(up2)    
    up7 = concatenate([up2, conv22], axis=4)
    conv71 = Conv3D(depth_cnn[1], (3, 3, 3), padding='same', name = 'conv5_1')(up7)
    conv71 = BatchNormalization()(conv71)
    conv71 = Activation('relu')(conv71)
    conc71 = concatenate([up7, conv71], axis=4)
    conv72 = Conv3D(depth_cnn[1], (3, 3, 3), padding='same', name = 'conv5_2')(conc71)
    conv72 = BatchNormalization()(conv72)
    conv72 = Activation('relu')(conv72)
    conv72 = concatenate([up7, conv72], axis=4)
    
    #decoder block3
    up3 = Permute((3,1,2,4))(conv72)
    up3 = TimeDistributed(Conv2DTranspose(depth_cnn[0], (2, 2), strides=(2, 2), padding='same', name = 'up3'))(up3)
    up3 = Permute((2,3,1,4))(up3)
    up8 = concatenate([up3, conv12], axis=4)
    conv81 = Conv3D(depth_cnn[0], (3, 3, 3), padding='same', name = 'conv6_1')(up8)
    conv81 = BatchNormalization()(conv81)
    conv81 = Activation('relu')(conv81)
    conc81 = concatenate([up8, conv81], axis=4)
    conv82 = Conv3D(depth_cnn[0], (3, 3, 3), padding='same', name = 'conv6_2')(conc81)
    conv82 = BatchNormalization()(conv82)
    conv82 = Activation('relu')(conv82)
    conc82 = concatenate([up8, conv82], axis=4)

    ##end of decoder block

    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid', name = 'final')(conc82)

    model = Model(inputs=[inputs], outputs=[conv10])
    
    model.compile(optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss = 'binary_crossentropy', metrics=[dice_coef])

    return model

class ConvBlock2D(Layer):
    def __init__(self, n_filters):
        super(ConvBlock2D, self).__init__()
        self.n_filters = n_filters
        self.conv1 = Conv2D(n_filters, (3,3), padding = 'same')
        self.bn1 = BatchNormalization()
        self.ac1 = Activation('relu')
        self.conv2 = Conv2D(n_filters, (3,3), padding = 'same')
        self.bn2 = BatchNormalization()
        self.ac2 = Activation('relu')

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.ac1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.ac2(x)
        return x
    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.n_filters)

class UNet2D_3x(Model):
    def __init__(self):
        super(UNet2D_3x, self).__init__()

        depth_cnn = [32, 64, 128, 256]
        
        self.conv1 = ConvBlock2D(depth_cnn[0])
        self.conv2 = ConvBlock2D(depth_cnn[1])
        self.conv3 = ConvBlock2D(depth_cnn[2])

        self.bottleneck = ConvBlock2D(depth_cnn[3])

        self.pool1 = MaxPooling2D((2,2))
        self.pool2 = MaxPooling2D((2,2))
        self.pool3 = MaxPooling2D((2,2))

        self.drop1 = SpatialDropout2D(0.1)
        self.drop2 = SpatialDropout2D(0.1)
        self.drop3 = SpatialDropout2D(0.1)
        
        self.up1 = Conv2DTranspose(depth_cnn[0], (2,2), strides = (2,2), padding = 'same')
        self.up2 = Conv2DTranspose(depth_cnn[1], (2,2), strides = (2,2), padding = 'same')
        self.up3 = Conv2DTranspose(depth_cnn[2], (2,2), strides = (2,2), padding = 'same')

        self.dec1 = ConvBlock2D(depth_cnn[0])
        self.dec2 = ConvBlock2D(depth_cnn[1])
        self.dec3 = ConvBlock2D(depth_cnn[2])

        self.one_conv = Conv2D(1, (1,1), activation = 'sigmoid')
    
    def call(self, input_tensor):

        x1 = self.conv1(input_tensor)
        x1_down = self.pool1(x1)
        x1_down = self.drop1(x1_down)

        x2 = self.conv2(x1_down)
        x2_down = self.pool2(x2)
        x2_down = self.drop2(x2_down)

        x3 = self.conv3(x2_down)
        x3_down = self.pool3(x3)
        x3_down = self.drop3(x3_down)

        base = self.bottleneck(x3_down)

        base_up = self.up3(base)
        c3 = concatenate([base_up, x3], axis = 3)
        d3 = self.dec3(c3)

        d3_up = self.up2(d3)
        c2 = concatenate([d3_up, x2], axis = 3)
        d2 = self.dec2(c2)

        d2_up = self.up1(d2)
        c1 = concatenate([d2_up, x1], axis = 3)
        d1 = self.dec1(c1)

        out = self.one_conv(d1)
        return out
    
    def build_graph(self, input_shape = (256, 256, 1)):
        input = Input(shape = input_shape)
        return Model(inputs = [input], outputs = self.call(input))


class UNet2D_4x(Model):
    def __init__(self):
        super(UNet2D_4x, self).__init__()

        depth_cnn = [32, 64, 128, 256, 512]
        
        self.conv1 = ConvBlock2D(depth_cnn[0])
        self.conv2 = ConvBlock2D(depth_cnn[1])
        self.conv3 = ConvBlock2D(depth_cnn[2])
        self.conv4 = ConvBlock2D(depth_cnn[3])

        self.bottleneck = ConvBlock2D(depth_cnn[4])

        self.pool1 = MaxPooling2D((2,2))
        self.pool2 = MaxPooling2D((2,2))
        self.pool3 = MaxPooling2D((2,2))
        self.pool4 = MaxPooling2D((2,2))

        self.drop1 = SpatialDropout2D(0.1)
        self.drop2 = SpatialDropout2D(0.1)
        self.drop3 = SpatialDropout2D(0.1)
        self.drop4 = SpatialDropout2D(0.1)
        
        self.up1 = Conv2DTranspose(depth_cnn[0], (2,2), strides = (2,2), padding = 'same')
        self.up2 = Conv2DTranspose(depth_cnn[1], (2,2), strides = (2,2), padding = 'same')
        self.up3 = Conv2DTranspose(depth_cnn[2], (2,2), strides = (2,2), padding = 'same')
        self.up4 = Conv2DTranspose(depth_cnn[3], (2,2), strides = (2,2), padding = 'same')

        self.dec1 = ConvBlock2D(depth_cnn[0])
        self.dec2 = ConvBlock2D(depth_cnn[1])
        self.dec3 = ConvBlock2D(depth_cnn[2])
        self.dec4 = ConvBlock2D(depth_cnn[3])

        self.one_conv = Conv2D(1, (1,1), activation = 'sigmoid')
    
    def call(self, input_tensor):

        x1 = self.conv1(input_tensor)
        x1_down = self.pool1(x1)
        x1_down = self.drop1(x1_down)

        x2 = self.conv2(x1_down)
        x2_down = self.pool2(x2)
        x2_down = self.drop2(x2_down)

        x3 = self.conv3(x2_down)
        x3_down = self.pool3(x3)
        x3_down = self.drop3(x3_down)

        x4 = self.conv4(x3_down)
        x4_down = self.pool4(x4)
        x4_down = self.drop4(x4_down)

        base = self.bottleneck(x4_down)

        base_up = self.up4(base)
        c4 = concatenate([base_up, x4], axis = 3)
        d4 = self.dec4(c4)

        d4_up = self.up3(d4)
        c3 = concatenate([d4_up, x3], axis = 3)
        d3 = self.dec3(c3)

        d3_up = self.up2(d3)
        c2 = concatenate([d3_up, x2], axis = 3)
        d2 = self.dec2(c2)

        d2_up = self.up1(d2)
        c1 = concatenate([d2_up, x1], axis = 3)
        d1 = self.dec1(c1)

        out = self.one_conv(d1)
        return out
    
    def build_graph(self, input_shape = (256, 256, 1)):
        input = Input(shape = input_shape)
        return Model(inputs = [input], outputs = self.call(input))

class ConvBlock3D(Layer):
    def __init__(self, n_filters):
        super(ConvBlock3D, self).__init__()
        self.n_filters = n_filters
        self.conv1 = Conv3D(n_filters, (3, 3, 3), padding = 'same')
        self.bn1 = BatchNormalization()
        self.ac1 = Activation('relu')
        self.conv2 = Conv3D(n_filters, (3, 3, 3), padding = 'same')
        self.bn2 = BatchNormalization()
        self.ac2 = Activation('relu')
    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.ac1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.ac2(x)
        return x
    #def compute_output_shape(self, input_shape):
    #    return (*input_shape[:-1], self.n_filters)

class Pool2D(Layer):
    def __init__(self):
        super(Pool2D, self).__init__()
        self.perm1 = Permute((3, 1, 2, 4))
        self.perm2 = Permute((2, 3, 1, 4))
        self.pool = MaxPooling2D(2, 2)
    def call(self, input_tensor):
        x = self.perm1(input_tensor)
        x = TimeDistributed(self.pool)(x)
        x = self.perm2(x)
        return x

class UpSample2D(Layer):
    def __init__(self, n_filters):
        super(UpSample2D, self).__init__()
        self.perm1 = Permute((3, 1, 2, 4))
        self.perm2 = Permute((2, 3, 1, 4))
        self.convt2d = Conv2DTranspose(n_filters, (2, 2), strides = (2, 2), padding = 'same')
    def call(self, input_tensor):
        x = self.perm1(input_tensor)
        x = TimeDistributed(self.convt2d)(x)
        x = self.perm2(x)
        return x

class UNet3D(Model):
    def __init__(self):
        super(UNet3D, self).__init__()
        depth_cnn = [32, 64, 128, 256]
        self.enc1 = ConvBlock3D(depth_cnn[0])
        self.enc2 = ConvBlock3D(depth_cnn[1])
        self.enc3 = ConvBlock3D(depth_cnn[2])

        self.drop1 = SpatialDropout3D(0.1)
        self.drop2 = SpatialDropout3D(0.1)
        self.drop3 = SpatialDropout3D(0.1)

        self.bottleneck = ConvBlock3D(depth_cnn[3])
        
        self.dec3 = ConvBlock3D(depth_cnn[2])
        self.dec2 = ConvBlock3D(depth_cnn[1])
        self.dec1 = ConvBlock3D(depth_cnn[0])

        self.pool1 = Pool2D()
        self.pool2 = Pool2D()
        self.pool3 = Pool2D()

        self.up3 = UpSample2D(depth_cnn[2])
        self.up2 = UpSample2D(depth_cnn[1])
        self.up1 = UpSample2D(depth_cnn[0])

        self.one_conv = Conv3D(1, (1, 1, 1), activation = 'sigmoid')
    
    def call(self, input_tensor):
        x1 = self.enc1(input_tensor)
        x1_down = self.pool1(x1)
        x1_down = self.drop1(x1_down)

        x2 = self.enc2(x1_down)
        x2_down = self.pool2(x2)
        x2_down = self.drop2(x2_down)

        x3 = self.enc3(x2_down)
        x3_down = self.pool3(x3)
        x3_down = self.drop3(x3_down)

        base = self.bottleneck(x3_down)

        base_up = self.up3(base)
        c3 = concatenate([base_up, x3], axis = 4)
        d3 = self.dec3(c3)

        d3_up = self.up2(d3)
        c2 = concatenate([d3_up, x2], axis = 4)
        d2 = self.dec2(c2)

        d2_up = self.up1(d2)
        c1 = concatenate([d2_up, x1], axis = 4)
        d1 = self.dec1(c1)

        seg = self.one_conv(d1)

        return seg
    
    def build_graph(self, input_shape = (256, 256, 8, 1)):
        input = Input(shape = input_shape)
        return Model(inputs = [input], outputs = self.call(input))

# --------------------------- UNet2D - Functional ------------------------------

def conv_block_2d(input, n_filters, name = None):
    x = Conv2D(n_filters, (3,3), padding = 'same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(n_filters, (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name = name)(x)
    return x

def get_UNet2Dx3(input_shape = (256, 256, 1)):
    input = Input(input_shape)
    depth_cnn = [32, 64, 128, 256]

    x1 = conv_block_2d(input, depth_cnn[0], name = 'x1')
    x1_down = MaxPooling2D((2,2), name = 'x1_down')(x1)
    x1_down = SpatialDropout2D(0.1)(x1_down)

    x2 = conv_block_2d(x1_down, depth_cnn[1], name = 'x2')
    x2_down = MaxPooling2D((2,2), name = 'x2_down')(x2)
    x2_down = SpatialDropout2D(0.1)(x2_down)

    x3 = conv_block_2d(x2_down, depth_cnn[2], name = 'x3')
    x3_down = MaxPooling2D((2,2), name = 'x3_down')(x3)
    x3_down = SpatialDropout2D(0.1)(x3_down)

    base = conv_block_2d(x3_down, depth_cnn[3])

    base_up = Conv2DTranspose(depth_cnn[2], (2,2), strides = (2,2), padding = 'same')(base)
    c3 = concatenate([base_up, x3], axis = 3)
    d3 = conv_block_2d(c3, depth_cnn[2])

    d3_up = Conv2DTranspose(depth_cnn[1], (2,2), strides = (2,2), padding = 'same')(d3)
    c2 = concatenate([d3_up, x2], axis = 3)
    d2 = conv_block_2d(c2, depth_cnn[1])

    d2_up = Conv2DTranspose(depth_cnn[0], (2,2), strides = (2,2), padding = 'same')(d2)
    c1 = concatenate([d2_up, x1], axis = 3)
    d1 = conv_block_2d(c1, depth_cnn[0])

    out = Conv2D(1, (1,1), activation = 'sigmoid')(d1)

    model = Model(inputs = [input], outputs = [out])
    return model

# --------------------------- UNet3D - Functional ------------------------------

# Block Definitions - Used in later models as well.

def conv_block_3d(input, n_filters):
    x = Conv3D(n_filters, (3, 3, 3), padding = 'same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(n_filters, (3, 3, 3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def pool_2d(input):
    x = Permute((3, 1, 2, 4))(input)
    x = TimeDistributed(MaxPooling2D(2, 2))(x)
    x = Permute((2, 3, 1, 4))(x)
    return x

def upsample_2d(input, n_filters):
    x = Permute((3, 1, 2, 4))(input)
    x = TimeDistributed(
        Conv2DTranspose(n_filters, (2, 2), strides = (2, 2), padding = 'same')
    )(x)
    x = Permute((2, 3, 1, 4))(x)
    return x

# ------------------------------------------------------------------------------

def get_UNet3D(input_shape = (256, 256, 8, 1)):
    depth_cnn = [32, 64, 128, 256]

    input = Input(input_shape)
    x1 = conv_block_3d(input, depth_cnn[0])
    x1_down = pool_2d(x1)
    x1_down = SpatialDropout3D(0.1)(x1_down)

    x2 = conv_block_3d(x1_down, depth_cnn[1])
    x2_down = pool_2d(x2)
    x2_down = SpatialDropout3D(0.1)(x2_down)

    x3 = conv_block_3d(x2_down, depth_cnn[2])
    x3_down = pool_2d(x3)
    x3_down = SpatialDropout3D(0.1)(x3_down)

    base = conv_block_3d(x3_down, depth_cnn[3])

    base_up = upsample_2d(base, depth_cnn[2])
    c3 = concatenate([base_up, x3], axis = 4)
    d3 = conv_block_3d(c3, depth_cnn[2])

    d3_up = upsample_2d(d3, depth_cnn[1])
    c2 = concatenate([d3_up, x2], axis = 4)
    d2 = conv_block_3d(c2, depth_cnn[1])

    d2_up = upsample_2d(d2, depth_cnn[0])
    c1 = concatenate([d2_up, x1], axis = 4)
    d1 = conv_block_3d(c1, depth_cnn[0])

    seg = Conv3D(1, (1, 1, 1), activation = 'sigmoid')(d1)

    model = Model(inputs = [input], outputs = [seg])
    return model

# ------------------------- Hybrid Functional Models ---------------------------

def get_model_fe(base_model, layers = None):
    if layers is None:
        layers = ['x1_down', 'x2_down', 'x3_down']

    layers_output = []

    for layer in layers:
        layers_output.append(base_model.get_layer(layer).output)

    return Model(inputs = [base_model.input], outputs = layers_output)

# Dependencies: 
# - conv_block_3d()
# - pool_2d()
# - upsample_2d()

def get_HybridUNet001(model_fe, input_shape = (256, 256, 8, 1)):
    '''
    Usage: 
    base_model = load_model('preferred type: UNet2Dx3 (functional)') 
    model_fe = get_model_fe(base_model)
    model = get_HybridUNet001(model_fe)
    '''
    depth_cnn = [32, 64, 128, 256]
    input = Input(input_shape)

    # Feature Extraction
    batch_input = Permute((3, 1, 2, 4))(input)

    outputs = []
    for out in model_fe.output:
        outputs.append(
            Permute((2, 3, 1, 4))(
                TimeDistributed(Model(model_fe.input, out))(batch_input)
            )
        )

    enc1_low, enc2_low, enc3_low = outputs

    x1 = conv_block_3d(input, depth_cnn[0])
    x1_down = pool_2d(x1)
    x1_down = SpatialDropout3D(0.1)(x1_down)

    x2 = concatenate([x1_down, enc1_low], axis = 4)
    x2 = conv_block_3d(x2, depth_cnn[1])
    x2_down = pool_2d(x2)
    x2_down = SpatialDropout3D(0.1)(x2_down)

    x3 = concatenate([x2_down, enc2_low], axis = 4)
    x3 = conv_block_3d(x3, depth_cnn[2])
    x3_down = pool_2d(x3)
    x3_down = SpatialDropout3D(0.1)(x3_down)

    base = concatenate([x3_down, enc3_low], axis = 4)
    base = conv_block_3d(base, depth_cnn[3])

    base_up = upsample_2d(base, depth_cnn[2])
    c3 = concatenate([base_up, x3], axis = 4)
    d3 = conv_block_3d(c3, depth_cnn[2])

    d3_up = upsample_2d(d3, depth_cnn[1])
    c2 = concatenate([d3_up, x2], axis = 4)
    d2 = conv_block_3d(c2, depth_cnn[1])

    d2_up = upsample_2d(d2, depth_cnn[0])
    c1 = concatenate([d2_up, x1], axis = 4)
    d1 = conv_block_3d(c1, depth_cnn[0])

    seg = Conv3D(1, (1, 1, 1), activation = 'sigmoid')(d1)

    model = Model(inputs = [input], outputs = [seg])

    return model

# --------------------------- R3DUNet - Functional -----------------------------

def dense_conv_block_3d(input, n_filters):
    x = Conv3D(n_filters, (3,3,3), padding = 'same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = concatenate([input, x], axis = 4)
    x = Conv3D(n_filters, (3,3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = concatenate([input, x], axis = 4)
    return x

def bneck_convLSTM(input, n_filters):
    x = BatchNormalization()(
        ConvLSTM2D(
            filters = n_filters, kernel_size = (3, 3), padding = 'same',
            return_sequences = True)(input))
    x = BatchNormalization()(
        ConvLSTM2D(
            filters = n_filters, kernel_size = (3, 3), padding = 'same',
            return_sequences = True)(x))
    x = BatchNormalization()(
        ConvLSTM2D(
            filters = n_filters, kernel_size = (3, 3), padding = 'same',
            return_sequences = True)(x))
    return x

# Dependencies Above: 
# - pool_2d
# - upsample_2d

def get_R3DDUNet(input_shape = (256, 256, 8, 1)):
    depth_cnn = [32, 64, 128, 256]

    input = Input(input_shape)
    x1 = dense_conv_block_3d(input, depth_cnn[0])
    x1_down = pool_2d(x1)
    x1_down = SpatialDropout3D(0.1)(x1_down)

    x2 = dense_conv_block_3d(x1_down, depth_cnn[1])
    x2_down = pool_2d(x2)
    x2_down = SpatialDropout3D(0.1)(x2_down)

    x3 = dense_conv_block_3d(x2_down, depth_cnn[2])
    x3_down = pool_2d(x3)
    x3_down = SpatialDropout3D(0.1)(x3_down)

    base = bneck_convLSTM(x3_down, depth_cnn[3])

    base_up = upsample_2d(base, depth_cnn[2])
    c3 = concatenate([base_up, x3], axis = 4)
    d3 = dense_conv_block_3d(c3, depth_cnn[2])

    d3_up = upsample_2d(d3, depth_cnn[1])
    c2 = concatenate([d3_up, x2], axis = 4)
    d2 = dense_conv_block_3d(c2, depth_cnn[1])

    d2_up = upsample_2d(d2, depth_cnn[0])
    c1 = concatenate([d2_up, x1], axis = 4)
    d1 = dense_conv_block_3d(c1, depth_cnn[0])

    seg = Conv3D(1, (1, 1, 1), activation = 'sigmoid')(d1)

    model = Model(inputs = [input], outputs = [seg])
    return model


# ------------------------ Hybrid R3DUNets - Functional ------------------------

# Dependencies:
# - get_model_fe(base_model)

def get_HybridRDUNet001(model_fe, input_shape = (256, 256, 8, 1)):
    '''
    Usage: 
    base_model = load_model('preferred type: dense_UNet2D_3x (functional)') 
    model_fe = get_model_fe(base_model)
    model = get_HybridRDUNet001(model_fe)
    '''
    depth_cnn = [32, 64, 128, 256]
    input = Input(input_shape)

    # Feature Extraction
    batch_input = Permute((3, 1, 2, 4))(input)

    outputs = []
    for out in model_fe.output:
        outputs.append(
            Permute((2, 3, 1, 4))(
                TimeDistributed(Model(model_fe.input, out))(batch_input)
            )
        )

    enc1_low, enc2_low, enc3_low = outputs

    x1 = dense_conv_block_3d(input, depth_cnn[0])
    x1_down = pool_2d(x1)
    x1_down = SpatialDropout3D(0.1)(x1_down)

    x2 = concatenate([x1_down, enc1_low], axis = 4)
    x2 = dense_conv_block_3d(x2, depth_cnn[1])
    x2_down = pool_2d(x2)
    x2_down = SpatialDropout3D(0.1)(x2_down)

    x3 = concatenate([x2_down, enc2_low], axis = 4)
    x3 = dense_conv_block_3d(x3, depth_cnn[2])
    x3_down = pool_2d(x3)
    x3_down = SpatialDropout3D(0.1)(x3_down)

    base = concatenate([x3_down, enc3_low], axis = 4)
    base = bneck_convLSTM(base, depth_cnn[3])

    base_up = upsample_2d(base, depth_cnn[2])
    c3 = concatenate([base_up, x3], axis = 4)
    d3 = dense_conv_block_3d(c3, depth_cnn[2])

    d3_up = upsample_2d(d3, depth_cnn[1])
    c2 = concatenate([d3_up, x2], axis = 4)
    d2 = dense_conv_block_3d(c2, depth_cnn[1])

    d2_up = upsample_2d(d2, depth_cnn[0])
    c1 = concatenate([d2_up, x1], axis = 4)
    d1 = dense_conv_block_3d(c1, depth_cnn[0])

    seg = Conv3D(1, (1, 1, 1), activation = 'sigmoid')(d1)

    model = Model(inputs = [input], outputs = [seg])
    return model

# ----------------------- MultiResUNet 2D - Functional -------------------------

def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    if(activation == None):
        return x

    x = Activation(activation, name=name)(x)

    return x

def MultiResBlock2D(U, inp, alpha = 1.67, name = None):
    W = alpha * U

    shortcut = inp
    shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
                         int(W*0.5), 1, 1, activation=None, padding='same')

    conv3x3 = conv2d_bn(inp, int(W*0.167), 3, 3,
                        activation='relu', padding='same')

    conv5x5 = conv2d_bn(conv3x3, int(W*0.333), 3, 3,
                        activation='relu', padding='same')

    conv7x7 = conv2d_bn(conv5x5, int(W*0.5), 3, 3,
                        activation='relu', padding='same')

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    out = BatchNormalization(axis=3)(out)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3, name = name)(out)

    return out

def ResPath2D(filters, length, inp):
    shortcut = inp
    shortcut = conv2d_bn(shortcut, filters, 1, 1,
                         activation=None, padding='same')

    out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    for i in range(length-1):

        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, 1, 1,
                             activation=None, padding='same')

        out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)

    return out

# --------------------------- MRUNet2Dx3 ---------------------------------------

def get_MRUNet2Dx3(height = 256, width = 256, n_channels = 1):
    inputs = Input((height, width, n_channels))

    enc1 = MultiResBlock2D(32, inputs, name = 'x1')
    pool1 = MaxPooling2D(pool_size=(2,2), name = 'x1_down')(enc1)
    enc1_res = ResPath2D(32, 3, enc1)

    enc2 = MultiResBlock2D(32*2, pool1, name = 'x2')
    pool2 = MaxPooling2D(pool_size=(2,2), name = 'x2_down')(enc2)
    enc2_res = ResPath2D(32*2, 2, enc2)

    enc3 = MultiResBlock2D(32*4, pool2, name = 'x3')
    pool3 = MaxPooling2D(pool_size=(2,2), name = 'x3_down')(enc3)
    enc3_res = ResPath2D(32*4, 1, enc3)

    base = MultiResBlock2D(32*8, pool3)

    base_up = concatenate([Conv2DTranspose(
        32*4, (2, 2), strides=(2, 2), padding='same')(base), enc3_res], axis=3)
    dec3 = MultiResBlock2D(32*4, base_up)

    dec3_up = concatenate([Conv2DTranspose(
        32*2, (2, 2), strides=(2, 2), padding='same')(dec3), enc2_res], axis=3)
    dec2 = MultiResBlock2D(32*2, dec3_up)

    dec2_up = concatenate([Conv2DTranspose(32, (2, 2), strides=(
        2, 2), padding='same')(dec2), enc1_res], axis=3)
    dec1 = MultiResBlock2D(32, dec2_up)

    out = conv2d_bn(dec1, 1, 1, 1, activation='sigmoid')

    model = Model(inputs = [inputs], outputs = [out])
    return model

# --------------------------- MRUNet2Dx4 ---------------------------------------

def get_MRUNet2Dx4(height = 256, width = 256, n_channels = 1):
    inputs = Input((height, width, n_channels))

    enc1 = MultiResBlock2D(32, inputs, name = 'x1')
    pool1 = MaxPooling2D(pool_size=(2,2), name = 'x1_down')(enc1)
    enc1_res = ResPath2D(32, 4, enc1)

    enc2 = MultiResBlock2D(32*2, pool1, name = 'x2')
    pool2 = MaxPooling2D(pool_size=(2,2), name = 'x2_down')(enc2)
    enc2_res = ResPath2D(32*2, 3, enc2)

    enc3 = MultiResBlock2D(32*4, pool2, name = 'x3')
    pool3 = MaxPooling2D(pool_size=(2,2), name = 'x3_down')(enc3)
    enc3_res = ResPath2D(32*4, 2, enc3)

    enc4 = MultiResBlock2D(32*8, pool3, name = 'x4')
    pool4 = MaxPooling2D(pool_size=(2,2), name = 'x4_down')(enc4)
    enc4_res = ResPath2D(32*8, 1, enc4)

    base = MultiResBlock2D(32*16, pool4)

    base_up = concatenate([Conv2DTranspose(
        32*8, (2, 2), strides=(2, 2), padding='same')(base), enc4_res], axis=3)
    dec4 = MultiResBlock2D(32*8, base_up)

    dec4_up = concatenate([Conv2DTranspose(
        32*4, (2, 2), strides=(2, 2), padding='same')(dec4), enc3_res], axis=3)
    dec3 = MultiResBlock2D(32*4, dec4_up)

    dec3_up = concatenate([Conv2DTranspose(
        32*2, (2, 2), strides=(2, 2), padding='same')(dec3), enc2_res], axis=3)
    dec2 = MultiResBlock2D(32*2, dec3_up)

    dec2_up = concatenate([Conv2DTranspose(32, (2, 2), strides=(
        2, 2), padding='same')(dec2), enc1_res], axis=3)
    dec1 = MultiResBlock2D(32, dec2_up)

    out = conv2d_bn(dec1, 1, 1, 1, activation='sigmoid')
    model = Model(inputs = [inputs], outputs = [out])
    return model

# --------------------------- MRUNet3D - f -------------------------------------

def conv3d_bn(x, filters, num_row, num_col, num_z, padding='same', strides=(1, 1, 1), activation='relu', name=None):
    x = Conv3D(filters, (num_row, num_col, num_z), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=4, scale=False)(x)

    if(activation==None):
        return x

    x = Activation(activation, name=name)(x)
    return x

def MultiResBlock3D(U, inp, alpha = 1.67):   
    W = alpha * U

    shortcut = inp

    shortcut = conv3d_bn(shortcut, int(W*0.167) + int(W*0.333) + int(W*0.5), 1, 1, 1, activation=None, padding='same')

    conv3x3 = conv3d_bn(inp, int(W*0.167), 3, 3, 3, activation='relu', padding='same')

    conv5x5 = conv3d_bn(conv3x3, int(W*0.333), 3, 3, 3, activation='relu', padding='same')

    conv7x7 = conv3d_bn(conv5x5, int(W*0.5), 3, 3, 3, activation='relu', padding='same')

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=4)
    out = BatchNormalization(axis=4)(out)
    
    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=4)(out)

    return out

def ResPath3D(filters, length, inp):
    shortcut = inp
    shortcut = conv3d_bn(
        shortcut, filters , 1, 1, 1, activation=None, padding='same')

    out = conv3d_bn(inp, filters, 3, 3, 3, activation='relu', padding='same')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=4)(out)

    for i in range(length-1):

        shortcut = out
        shortcut = conv3d_bn(
            shortcut, filters , 1, 1, 1, activation=None, padding='same')

        out = conv3d_bn(out, filters, 3, 3, 3, activation='relu', padding='same')        

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=4)(out)

    return out

# ------------------------------------------------------------------------------

# Dependencies: 
#  - MultiResBlock3D
#    - conv3d_bn
#  - ResPath3D
#  - pool_2d
#  - upsample_2d


def get_MRUNet3D(height = 256, width = 256, depth = 8, n_channels = 1):
    depth_cnn = [32, 64, 128, 256]
    
    inputs = Input((height, width, depth, n_channels))

    x1 = MultiResBlock3D(depth_cnn[0], inputs)
    x1_down = pool_2d(x1)
    x1_res = ResPath3D(depth_cnn[0], 3, x1)

    x2 = MultiResBlock3D(depth_cnn[1], x1_down)
    x2_down = pool_2d(x2)
    x2_res = ResPath3D(depth_cnn[1], 2, x2)

    x3 = MultiResBlock3D(depth_cnn[2], x2_down)
    x3_down = pool_2d(x3)
    x3_res = ResPath3D(depth_cnn[2], 2, x3)

    base = MultiResBlock3D(depth_cnn[3], x3_down)

    base_up = upsample_2d(base, depth_cnn[2])
    c3 = concatenate([base_up, x3_res], axis = 4)
    d3 = MultiResBlock3D(depth_cnn[2], c3)

    d3_up = upsample_2d(d3, depth_cnn[1])
    c2 = concatenate([d3_up, x2_res], axis = 4)
    d2 = MultiResBlock3D(depth_cnn[1], c2)
 
    d2_up = upsample_2d(d2, depth_cnn[0])
    c1 = concatenate([d2_up, x1_res], axis = 4)
    d1 = MultiResBlock3D(depth_cnn[0], c1)

    out = conv3d_bn(d1 , 1, 1, 1, 1, activation='sigmoid')

    model = Model(inputs = [inputs], outputs = [out])
    return model

# --------------------------- Hybrid MRUNets -----------------------------------

def get_HybridMRUNet3D_001(model_fe, input_shape = (256, 256, 8, 1)):
    '''
    Usage: 
    base_model = load_model('preferred type: dense_UNet2D_3x (functional)') 
    model_fe = get_model_fe(base_model)
    model = get_HybridRDUNet001(model_fe)
    '''
    depth_cnn = [32, 64, 128, 256]

    input = Input(input_shape)

    # Feature Extraction

    batch_input = Permute((3, 1, 2, 4))(input)

    outputs = []

    for out in model_fe.output:
        outputs.append(
            Permute((2, 3, 1, 4))(
                TimeDistributed(Model(model_fe.input, out))(batch_input)
            )
        )

    enc1_low, enc2_low, enc3_low = outputs

    x1 = MultiResBlock3D(depth_cnn[0], input)
    x1_down = pool_2d(x1)
    x1_res = ResPath3D(depth_cnn[0], 3, x1)

    x2 = concatenate([x1_down, enc1_low], axis = 4)
    x2 = MultiResBlock3D(depth_cnn[1], x2)
    x2_down = pool_2d(x2)
    x2_res = ResPath3D(depth_cnn[1], 2, x2)

    x3 = concatenate([x2_down, enc2_low], axis = 4)
    x3 = MultiResBlock3D(depth_cnn[2], x3)
    x3_down = pool_2d(x3)
    x3_res = ResPath3D(depth_cnn[2], 1, x3)

    base = MultiResBlock3D(depth_cnn[3], x3_down)

    base_up = upsample_2d(base, depth_cnn[2])
    c3 = concatenate([base_up, x3_res], axis = 4)
    d3 = MultiResBlock3D(depth_cnn[2], c3)

    d3_up = upsample_2d(d3, depth_cnn[1])
    c2 = concatenate([d3_up, x2_res], axis = 4)
    d2 = MultiResBlock3D(depth_cnn[1], c2)

    d2_up = upsample_2d(d2, depth_cnn[0])
    c1 = concatenate([d2_up, x1_res], axis = 4)
    d1 = MultiResBlock3D(depth_cnn[0], c1)

    out = conv3d_bn(d1 , 1, 1, 1, 1, activation='sigmoid')

    model = Model(inputs = [input], outputs = [out])

    return model


# --------------------------- Hybrid Models ------------------------------------


class Extract2DFeat(Layer):
    def __init__(self, base_model):
        super(Extract2DFeat, self).__init__()
        self.enc1 = base_model.conv1
        self.enc2 = base_model.conv2
        self.enc3 = base_model.conv3
        self.pool1 = base_model.pool1
        self.pool2 = base_model.pool2
        self.pool3 = base_model.pool3
        self.perm2d = Permute((3, 1, 2, 4))
        self.perm3d = Permute((2, 3, 1, 4))
    def call(self, input_tensor):
        input = self.perm2d(input_tensor)
        x1 = TimeDistributed(self.enc1)(input)
        x1_low = TimeDistributed(self.pool2)(x1)
        x2 = TimeDistributed(self.enc2)(x1_low)
        x2_low = TimeDistributed(self.pool2)(x2)
        x3 = TimeDistributed(self.enc3)(x2_low)
        x3_low = TimeDistributed(self.pool3)(x3)
        x1 = self.perm3d(x1)
        x1_low = self.perm3d(x1_low)
        x2 = self.perm3d(x2)
        x2_low = self.perm3d(x2_low)
        x3 = self.perm3d(x3)
        x3_low = self.perm3d(x3_low)

        return {
            'enc1_high': x1,
            'enc1_low': x1_low,
            'enc2_high': x2,
            'enc2_low': x2_low,
            'enc3_high': x3,
            'enc3_low': x3_low,
        }

class HybridUNet001(Model):
    def __init__(self, base_model):
        super(HybridUNet001, self).__init__()

        depth_cnn = [32, 64, 128, 256]

        self.feat_model = Extract2DFeat(base_model)

        self.enc1 = ConvBlock3D(depth_cnn[0])
        self.enc2 = ConvBlock3D(depth_cnn[1])
        self.enc3 = ConvBlock3D(depth_cnn[2])

        self.drop1 = SpatialDropout3D(0.1)
        self.drop2 = SpatialDropout3D(0.1)
        self.drop3 = SpatialDropout3D(0.1)

        self.bottleneck = ConvBlock3D(depth_cnn[3])

        self.dec3 = ConvBlock3D(depth_cnn[2])
        self.dec2 = ConvBlock3D(depth_cnn[1])
        self.dec1 = ConvBlock3D(depth_cnn[0])

        self.pool1 = Pool2D()
        self.pool2 = Pool2D()
        self.pool3 = Pool2D()

        self.up3 = UpSample2D(depth_cnn[2])
        self.up2 = UpSample2D(depth_cnn[1])
        self.up1 = UpSample2D(depth_cnn[0])

        self.one_conv = Conv3D(1, (1, 1, 1), activation = 'sigmoid')
    
    def call(self, input_tensor):
        feats = self.feat_model(input_tensor)

        enc1_low = feats['enc1_low']
        enc2_low = feats['enc2_low']
        enc3_low = feats['enc3_low']
        
        x1 = self.enc1(input_tensor)
        x1_down = self.pool1(x1)
        x1_down = self.drop1(x1_down)

        x2 = concatenate([x1_down, enc1_low], axis = 4)
        x2 = self.enc2(x2)
        x2_down = self.pool2(x2)
        x2_down = self.drop2(x2_down)

        x3 = concatenate([x2_down, enc2_low], axis = 4)
        x3 = self.enc3(x3)
        x3_down = self.pool3(x3)
        x3_down = self.drop3(x3_down)

        base = concatenate([x3_down, enc3_low], axis = 4)
        base = self.bottleneck(base)

        base_up = self.up3(base)
        c3 = concatenate([base_up, x3], axis = 4)
        d3 = self.dec3(c3)

        d3_up = self.up2(d3)
        c2 = concatenate([d3_up, x2], axis = 4)
        d2 = self.dec2(c2)

        d2_up = self.up1(d2)
        c1 = concatenate([d2_up, x1], axis = 4)
        d1 = self.dec1(c1)

        out = self.one_conv(d1)
        return out
    
    def build_graph(self, input_shape = (256, 256, 8, 1)):
        input = Input(shape = input_shape)
        #input_tensor = tf.random.uniform(input_shape)
        return Model(inputs = [input], outputs = self.call(input))

class HybridUNet002(Model):
    def __init__(self, base_model):
        super(HybridUNet002, self).__init__()

        depth_cnn = [32, 64, 128, 256]

        self.feat_model = Extract2DFeat(base_model)

        self.enc1 = ConvBlock3D(depth_cnn[0])
        self.enc2 = ConvBlock3D(depth_cnn[1])
        self.enc3 = ConvBlock3D(depth_cnn[2])

        self.drop1 = SpatialDropout3D(0.1)
        self.drop2 = SpatialDropout3D(0.1)
        self.drop3 = SpatialDropout3D(0.1)

        self.bottleneck = ConvBlock3D(depth_cnn[3])

        self.dec3 = ConvBlock3D(depth_cnn[2])
        self.dec2 = ConvBlock3D(depth_cnn[1])
        self.dec1 = ConvBlock3D(depth_cnn[0])

        self.pool1 = Pool2D()
        self.pool2 = Pool2D()
        self.pool3 = Pool2D()

        self.up3 = UpSample2D(depth_cnn[2])
        self.up2 = UpSample2D(depth_cnn[1])
        self.up1 = UpSample2D(depth_cnn[0])

        self.one_conv = Conv3D(1, (1, 1, 1), activation = 'sigmoid')
    
    def call(self, input_tensor):
        feats = self.feat_model(input_tensor)
        enc1_low = feats['enc1_low']
        enc2_low = feats['enc2_low']
        enc3_low = feats['enc3_low']
        enc1_high = feats['enc1_high']
        enc2_high = feats['enc2_high']
        enc3_high = feats['enc3_high']
                
        x1 = self.enc1(input_tensor)
        x1_down = self.pool1(x1)
        x1_down = self.drop1(x1_down)

        x2 = concatenate([x1_down, enc1_low], axis = 4)
        x2 = self.enc2(x2)
        x2_down = self.pool2(x2)
        x2_down = self.drop2(x2_down)

        x3 = concatenate([x2_down, enc2_low], axis = 4)
        x3 = self.enc3(x3)
        x3_down = self.pool3(x3)
        x3_down = self.drop3(x3_down)

        base = concatenate([x3_down, enc3_low], axis = 4)
        base = self.bottleneck(base)

        base_up = self.up3(base)
        c3 = concatenate([base_up, enc3_high], axis = 4)
        d3 = self.dec3(c3)

        d3_up = self.up2(d3)
        c2 = concatenate([d3_up, enc2_high], axis = 4)
        d2 = self.dec2(c2)

        d2_up = self.up1(d2)
        c1 = concatenate([d2_up, enc1_high], axis = 4)
        d1 = self.dec1(c1)

        out = self.one_conv(d1)
        return out
    
    def build_graph(self, input_shape = (256, 256, 8, 1)):
        input = Input(shape = input_shape)
        #input_tensor = tf.random.uniform(input_shape)
        return Model(inputs = [input], outputs = self.call(input))

class HybridUNet003(Model):
    def __init__(self, base_model):
        super(HybridUNet003, self).__init__()

        depth_cnn = [32, 64, 128, 256]

        self.feat_model = Extract2DFeat(base_model)

        self.enc1 = ConvBlock3D(depth_cnn[0])
        self.enc2 = ConvBlock3D(depth_cnn[1])
        self.enc3 = ConvBlock3D(depth_cnn[2])

        self.catconv1 = ConvBlock3D(depth_cnn[0])
        self.catconv2 = ConvBlock3D(depth_cnn[1])
        self.catconv3 = ConvBlock3D(depth_cnn[2])

        self.drop1 = SpatialDropout3D(0.1)
        self.drop2 = SpatialDropout3D(0.1)
        self.drop3 = SpatialDropout3D(0.1)

        self.bottleneck = ConvBlock3D(depth_cnn[3])

        self.dec3 = ConvBlock3D(depth_cnn[2])
        self.dec2 = ConvBlock3D(depth_cnn[1])
        self.dec1 = ConvBlock3D(depth_cnn[0])

        self.pool1 = Pool2D()
        self.pool2 = Pool2D()
        self.pool3 = Pool2D()

        self.up3 = UpSample2D(depth_cnn[2])
        self.up2 = UpSample2D(depth_cnn[1])
        self.up1 = UpSample2D(depth_cnn[0])

        self.one_conv = Conv3D(1, (1, 1, 1), activation = 'sigmoid')
    
    def call(self, input_tensor):
        feats = self.feat_model(input_tensor)

        enc1_low = feats['enc1_low']
        enc2_low = feats['enc2_low']
        enc3_low = feats['enc3_low']
        enc1_high = feats['enc1_high']
        enc2_high = feats['enc2_high']
        enc3_high = feats['enc3_high']

        x1 = self.enc1(input_tensor)
        x1_down = self.pool1(x1)
        x1_down = self.drop1(x1_down)

        x2 = concatenate([x1_down, enc1_low], axis = 4)
        x2 = self.enc2(x2)
        x2_down = self.pool2(x2)
        x2_down = self.drop2(x2_down)

        x3 = concatenate([x2_down, enc2_low], axis = 4)
        x3 = self.enc3(x3)
        x3_down = self.pool3(x3)
        x3_down = self.drop3(x3_down)

        base = concatenate([x3_down, enc3_low], axis = 4)
        base = self.bottleneck(base)

        base_up = self.up3(base)

        cc1 = concatenate([x1, enc1_high], axis = 4)
        cc1 = self.catconv1(cc1)

        cc2 = concatenate([x2, enc2_high], axis = 4)
        cc2 = self.catconv2(cc2)

        cc3 = concatenate([x3, enc3_high], axis = 4)
        cc3 = self.catconv3(cc3)

        c3 = concatenate([base_up, cc3], axis = 4)
        d3 = self.dec3(c3)

        d3_up = self.up2(d3)
        c2 = concatenate([d3_up, cc2], axis = 4)
        d2 = self.dec2(c2)

        d2_up = self.up1(d2)
        c1 = concatenate([d2_up, cc1], axis = 4)
        d1 = self.dec1(c1)

        seg = self.one_conv(d1)
        return seg
    
    def build_graph(self, input_shape = (256, 256, 8, 1)):
        input = Input(shape = input_shape)
        return Model(inputs = [input], outputs = self.call(input))


# --------------------------- Hybrid DUNets ------------------------------------

class DenseConvBlock2D(Layer):
    def __init__(self, n_filters):
        super(DenseConvBlock2D, self).__init__()
        self.n_filters = n_filters
        self.conv1 = Conv2D(n_filters, (3,3), padding = 'same')
        self.bn1 = BatchNormalization()
        self.ac1 = Activation('relu')
        self.conv2 = Conv2D(n_filters, (3,3), padding = 'same')
        self.bn2 = BatchNormalization()
        self.ac2 = Activation('relu')

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.ac1(x)
        x = concatenate([input_tensor, x], axis = 3)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.ac2(x)
        x = concatenate([input_tensor, x], axis = 3)
        return x
    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], (self.n_filters + input_shape[-1]))
    #def compute_output_shape(self, input_shape):
    #    if input_shape[-1] is None:
    #        output_shape = (*input_shape[:-1], (self.n_filters + 1))
    #   else:
    #        output_shape =  (*input_shape[:-1], (self.n_filters + input_shape[-1]))
    #    return output_shape

class DenseUNet2D_4x(Model):
    def __init__(self):
        super(DenseUNet2D_4x, self).__init__()

        depth_cnn = [32, 64, 128, 256, 512]
        
        self.conv1 = DenseConvBlock2D(depth_cnn[0])
        self.conv2 = DenseConvBlock2D(depth_cnn[1])
        self.conv3 = DenseConvBlock2D(depth_cnn[2])
        self.conv4 = DenseConvBlock2D(depth_cnn[3])

        self.bottleneck = DenseConvBlock2D(depth_cnn[4])

        self.pool1 = MaxPooling2D((2,2))
        self.pool2 = MaxPooling2D((2,2))
        self.pool3 = MaxPooling2D((2,2))
        self.pool4 = MaxPooling2D((2,2))

        self.drop1 = SpatialDropout2D(0.1)
        self.drop2 = SpatialDropout2D(0.1)
        self.drop3 = SpatialDropout2D(0.1)
        self.drop4 = SpatialDropout2D(0.1)
        
        self.up1 = Conv2DTranspose(depth_cnn[0], (2,2), strides = (2,2), padding = 'same')
        self.up2 = Conv2DTranspose(depth_cnn[1], (2,2), strides = (2,2), padding = 'same')
        self.up3 = Conv2DTranspose(depth_cnn[2], (2,2), strides = (2,2), padding = 'same')
        self.up4 = Conv2DTranspose(depth_cnn[3], (2,2), strides = (2,2), padding = 'same')

        self.dec1 = DenseConvBlock2D(depth_cnn[0])
        self.dec2 = DenseConvBlock2D(depth_cnn[1])
        self.dec3 = DenseConvBlock2D(depth_cnn[2])
        self.dec4 = DenseConvBlock2D(depth_cnn[3])

        self.one_conv = Conv2D(1, (1,1), activation = 'sigmoid')
    
    def call(self, input_tensor):

        x1 = self.conv1(input_tensor)
        x1_down = self.pool1(x1)
        x1_down = self.drop1(x1_down)

        x2 = self.conv2(x1_down)
        x2_down = self.pool2(x2)
        x2_down = self.drop2(x2_down)

        x3 = self.conv3(x2_down)
        x3_down = self.pool3(x3)
        x3_down = self.drop3(x3_down)

        x4 = self.conv4(x3_down)
        x4_down = self.pool4(x4)
        x4_down = self.drop4(x4_down)

        base = self.bottleneck(x4_down)

        base_up = self.up4(base)
        c4 = concatenate([base_up, x4], axis = 3)
        d4 = self.dec4(c4)

        d4_up = self.up3(d4)
        c3 = concatenate([d4_up, x3], axis = 3)
        d3 = self.dec3(c3)

        d3_up = self.up2(d3)
        c2 = concatenate([d3_up, x2], axis = 3)
        d2 = self.dec2(c2)

        d2_up = self.up1(d2)
        c1 = concatenate([d2_up, x1], axis = 3)
        d1 = self.dec1(c1)

        out = self.one_conv(d1)
        return out
    
    def build_graph(self, input_shape = (256, 256, 1)):
        input = Input(shape = input_shape)
        return Model(inputs = [input], outputs = self.call(input))

class DenseUNet2D_4x_NB(Model):
    def __init__(self):
        super(DenseUNet2D_4x_NB, self).__init__()

        depth_cnn = [32, 64, 128, 256, 512]

        # Dense Conv Block 1
        self.conv1_1 = Conv2D(depth_cnn[0], (3, 3), padding = 'same')
        self.bn1_1 = BatchNormalization()
        self.ac1_1 = Activation('relu')
        self.conv1_2 = Conv2D(depth_cnn[0], (3, 3), padding = 'same')
        self.bn1_2 = BatchNormalization()
        self.ac1_2 = Activation('relu')

        self.pool1 = MaxPooling2D((2,2))

        # Dense Conv Block 2
        self.conv2_1 = Conv2D(depth_cnn[1], (3, 3), padding = 'same')
        self.bn2_1 = BatchNormalization()
        self.ac2_1 = Activation('relu')
        self.conv2_2 = Conv2D(depth_cnn[1], (3, 3), padding = 'same')
        self.bn2_2 = BatchNormalization()
        self.ac2_2 = Activation('relu')

        self.pool2 = MaxPooling2D((2,2))

        # Dense Conv Block 3
        self.conv3_1 = Conv2D(depth_cnn[2], (3, 3), padding = 'same')
        self.bn3_1 = BatchNormalization()
        self.ac3_1 = Activation('relu')
        self.conv3_2 = Conv2D(depth_cnn[2], (3, 3), padding = 'same')
        self.bn3_2 = BatchNormalization()
        self.ac3_2 = Activation('relu')

        self.pool3 = MaxPooling2D((2,2))

        # Dense Conv Block 4
        self.conv4_1 = Conv2D(depth_cnn[3], (3, 3), padding = 'same')
        self.bn4_1 = BatchNormalization()
        self.ac4_1 = Activation('relu')
        self.conv4_2 = Conv2D(depth_cnn[3], (3, 3), padding = 'same')
        self.bn4_2 = BatchNormalization()
        self.ac4_2 = Activation('relu')

        self.pool4 = MaxPooling2D((2,2))

        # Dropouts

        self.drop1 = SpatialDropout2D(0.1)
        self.drop2 = SpatialDropout2D(0.1)
        self.drop3 = SpatialDropout2D(0.1)
        self.drop4 = SpatialDropout2D(0.1)

        # Bottleneck Block

        self.conv5_1 = Conv2D(depth_cnn[4], (3, 3), padding = 'same')
        self.bn5_1 = BatchNormalization()
        self.ac5_1 = Activation('relu')
        self.conv5_2 = Conv2D(depth_cnn[4], (3, 3), padding = 'same')
        self.bn5_2 = BatchNormalization()
        self.ac5_2 = Activation('relu')

        # Upsamplers

        self.up1 = Conv2DTranspose(depth_cnn[0], (2,2), strides = (2,2), padding = 'same')
        self.up2 = Conv2DTranspose(depth_cnn[1], (2,2), strides = (2,2), padding = 'same')
        self.up3 = Conv2DTranspose(depth_cnn[2], (2,2), strides = (2,2), padding = 'same')
        self.up4 = Conv2DTranspose(depth_cnn[3], (2,2), strides = (2,2), padding = 'same')

        # Decoder Conv Block 

        self.conv6_1 = Conv2D(depth_cnn[3], (3, 3), padding = 'same')
        self.bn6_1 = BatchNormalization()
        self.ac6_1 = Activation('relu')
        self.conv6_2 = Conv2D(depth_cnn[3], (3, 3), padding = 'same')
        self.bn6_2 = BatchNormalization()
        self.ac6_2 = Activation('relu')

        self.conv7_1 = Conv2D(depth_cnn[2], (3, 3), padding = 'same')
        self.bn7_1 = BatchNormalization()
        self.ac7_1 = Activation('relu')
        self.conv7_2 = Conv2D(depth_cnn[2], (3, 3), padding = 'same')
        self.bn7_2 = BatchNormalization()
        self.ac7_2 = Activation('relu')

        self.conv8_1 = Conv2D(depth_cnn[1], (3, 3), padding = 'same')
        self.bn8_1 = BatchNormalization()
        self.ac8_1 = Activation('relu')
        self.conv8_2 = Conv2D(depth_cnn[1], (3, 3), padding = 'same')
        self.bn8_2 = BatchNormalization()
        self.ac8_2 = Activation('relu')

        self.conv9_1 = Conv2D(depth_cnn[0], (3, 3), padding = 'same')
        self.bn9_1 = BatchNormalization()
        self.ac9_1 = Activation('relu')
        self.conv9_2 = Conv2D(depth_cnn[0], (3, 3), padding = 'same')
        self.bn9_2 = BatchNormalization()
        self.ac9_2 = Activation('relu')

        self.one_conv = Conv2D(1, (1,1), activation = 'sigmoid')

    def call(self, input_tensor):

        x1 = self.conv1_1(input_tensor)
        x1 = self.bn1_1(x1)
        x1 = self.ac1_1(x1)
        x1 = concatenate([input_tensor, x1], axis = 3)

        x1 = self.conv1_2(x1)
        x1 = self.bn1_2(x1)
        x1 = self.ac1_2(x1)
        x1 = concatenate([input_tensor, x1], axis = 3)

        x1_down = self.pool1(x1)
        x1_down = self.drop1(x1_down)

        x2 = self.conv2_1(x1_down)
        x2 = self.bn2_1(x2)
        x2 = self.ac2_1(x2)
        x2 = concatenate([x1_down, x2], axis = 3)
 
        x2 = self.conv2_2(x2)
        x2 = self.bn2_2(x2)
        x2 = self.ac2_2(x2)
        x2 = concatenate([x1_down, x2], axis = 3)

        x2_down = self.pool2(x2)
        x2_down = self.drop2(x2_down)

        x3 = self.conv3_1(x2_down)
        x3 = self.bn3_1(x3)
        x3 = self.ac3_1(x3)
        x3 = concatenate([x2_down, x3], axis = 3)

        x3 = self.conv3_2(x3)
        x3 = self.bn3_2(x3)
        x3 = self.ac3_2(x3)
        x3 = concatenate([x2_down, x3], axis = 3)

        x3_down = self.pool3(x3)
        x3_down = self.drop3(x3_down)

        x4 = self.conv4_1(x3_down)
        x4 = self.bn4_1(x4)
        x4 = self.ac4_1(x4)
        x4 = concatenate([x3_down, x4], axis = 3)
        x4 = self.conv4_2(x4)
        x4 = self.bn4_2(x4)
        x4 = self.ac4_2(x4)
        x4 = concatenate([x3_down, x4], axis = 3)

        x4_down = self.pool4(x4)
        x4_down = self.drop4(x4_down)

        base = self.conv5_1(x4_down)
        base = self.bn5_1(base)
        base = self.ac5_1(base)
        base = concatenate([x4_down, base], axis = 3)
        base = self.conv5_2(base)
        base = self.bn5_2(base)
        base = self.ac5_2(base)
        base = concatenate([x4_down, base], axis = 3)

        base_up = self.up4(base)
        c4 = concatenate([base_up, x4], axis = 3)
        d4 = self.conv6_1(c4)
        d4 = self.bn6_1(d4)
        d4 = self.ac6_1(d4)
        d4 = concatenate([c4, d4], axis = 3)
        d4 = self.conv6_2(d4)
        d4 = self.bn6_2(d4)
        d4 = self.ac6_2(d4)
        d4 = concatenate([c4, d4], axis = 3)

        d4_up = self.up3(d4)
        c3 = concatenate([d4_up, x3], axis = 3)
        d3 = self.conv7_1(c3)
        d3 = self.bn7_1(d3)
        d3 = self.ac7_1(d3)
        d3 = concatenate([c3, d3], axis = 3)
        d3 = self.conv7_2(d3)
        d3 = self.bn7_2(d3)
        d3 = self.ac7_2(d3)
        d3 = concatenate([c3, d3], axis = 3)

        d3_up = self.up2(d3)
        c2 = concatenate([d3_up, x2], axis = 3)
        d2 = self.conv8_1(c2)
        d2 = self.bn8_1(d2)
        d2 = self.ac8_1(d2)
        d2 = concatenate([c2, d2], axis = 3)
        d2 = self.conv8_2(d2)
        d2 = self.bn8_2(d2)
        d2 = self.ac8_2(d2)
        d2 = concatenate([c2, d2], axis = 3)

        d2_up = self.up1(d2)
        c1 = concatenate([d2_up, x1], axis = 3)
        d1 = self.conv9_1(c1)
        d1 = self.bn9_1(d1)
        d1 = self.ac9_1(d1)
        d1 = concatenate([c1, d1], axis = 3)
        d1 = self.conv9_2(d1)
        d1 = self.bn9_2(d1)
        d1 = self.ac9_2(d1)
        d1 = concatenate([c1, d1], axis = 3)

        out = self.one_conv(d1)
        return out
    
    def build_graph(self, input_shape = (256, 256, 1)):
        input = Input(shape = input_shape)
        return Model(inputs = [input], outputs = self.call(input))

# ---------------------------  RDUNet SubClass ---------------------------------

class DenseConvBlock3D(Layer):
    def __init__(self, n_filters):
        super(DenseConvBlock3D, self).__init__()
        self.n_filters = n_filters
        self.conv1 = Conv3D(n_filters, (3,3,3), padding = 'same')
        self.bn1 = BatchNormalization()
        self.ac1 = Activation('relu')
        self.conv2 = Conv3D(n_filters, (3,3,3), padding = 'same')
        self.bn2 = BatchNormalization()
        self.ac2 = Activation('relu')

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.ac1(x)
        x = concatenate([input_tensor, x], axis = 4)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.ac2(x)
        x = concatenate([input_tensor, x], axis = 4)
        return x
    #def compute_output_shape(self, input_shape):
    #    return (*input_shape[:-1], (self.n_filters + input_shape[-1]))

class BNeckConvLSTM(Layer):
    def __init__(self, n_filters):
        super(BNeckConvLSTM, self).__init__()
        self.clstm1 = ConvLSTM2D(filters = n_filters, kernel_size = (3, 3), padding = 'same', return_sequences = True)
        self.clstm2 = ConvLSTM2D(filters = n_filters, kernel_size = (3, 3), padding = 'same', return_sequences = True)
        self.clstm3 = ConvLSTM2D(filters = n_filters, kernel_size = (3, 3), padding = 'same', return_sequences = True)
    def call(self, input_tensor):
        x = BatchNormalization()(self.clstm1(input_tensor))
        x = BatchNormalization()(self.clstm2(x))
        x = BatchNormalization()(self.clstm3(x))
        return x

class R3DDUNet(Model):
    def __init__(self):
        super(R3DDUNet, self).__init__()
        depth_cnn = [32, 64, 128, 256]

        self.enc1 = DenseConvBlock3D(depth_cnn[0])
        self.enc2 = DenseConvBlock3D(depth_cnn[1])
        self.enc3 = DenseConvBlock3D(depth_cnn[2])

        self.drop1 = SpatialDropout3D(0.1)
        self.drop2 = SpatialDropout3D(0.1)
        self.drop3 = SpatialDropout3D(0.1)

        self.pool1 = Pool2D()
        self.pool2 = Pool2D()
        self.pool3 = Pool2D()

        self.bottleneck = BNeckConvLSTM(depth_cnn[3])

        self.dec3 = DenseConvBlock3D(depth_cnn[2])
        self.dec2 = DenseConvBlock3D(depth_cnn[1])
        self.dec1 = DenseConvBlock3D(depth_cnn[0])

        self.up3 = UpSample2D(depth_cnn[2])
        self.up2 = UpSample2D(depth_cnn[1])
        self.up1 = UpSample2D(depth_cnn[0])

        self.one_conv = Conv3D(1, (1,1,1), activation = 'sigmoid')

    def call(self, input_tensor):
        x1 = self.enc1(input_tensor)
        x1_down = self.pool1(x1)
        x1_down = self.drop1(x1_down)

        x2 = self.enc2(x1_down)
        x2_down = self.pool2(x2)
        x2_down = self.drop2(x2_down)

        x3 = self.enc3(x2_down)
        x3_down = self.pool3(x3)
        x3_down = self.drop3(x3_down)

        base = self.bottleneck(x3_down)

        base_up = self.up3(base)
        c3 = concatenate([base_up, x3], axis = 4)
        d3 = self.dec3(c3)

        d3_up = self.up2(d3)
        c2 = concatenate([d3_up, x2], axis = 4)
        d2 = self.dec2(c2)

        d2_up = self.up1(d2)
        c1 = concatenate([d2_up, x1], axis = 4)
        d1 = self.dec1(c1)

        seg = self.one_conv(d1)
        return seg
    
    def build_graph(self, input_shape = (256, 256, 8, 1)):
        input = Input(shape = input_shape)
        return Model(inputs = [input], outputs = self.call(input))

# ---------------------------- Hybrid RDUNet -----------------------------------

class HybridRDUNet_001(Model):
    def __init__(self, base_model):
        super(HybridRDUNet_001, self).__init__()
        depth_cnn = [32, 64, 128, 256]

        self.feat_model = Extract2DFeat(base_model)

        self.enc1 = DenseConvBlock3D(depth_cnn[0])
        self.enc2 = DenseConvBlock3D(depth_cnn[1])
        self.enc3 = DenseConvBlock3D(depth_cnn[2])

        self.drop1 = SpatialDropout3D(0.1)
        self.drop2 = SpatialDropout3D(0.1)
        self.drop3 = SpatialDropout3D(0.1)

        self.pool1 = Pool2D()
        self.pool2 = Pool2D()
        self.pool3 = Pool2D()

        self.bottleneck = BNeckConvLSTM(depth_cnn[3])

        self.dec3 = DenseConvBlock3D(depth_cnn[2])
        self.dec2 = DenseConvBlock3D(depth_cnn[1])
        self.dec1 = DenseConvBlock3D(depth_cnn[0])

        self.up3 = UpSample2D(depth_cnn[2])
        self.up2 = UpSample2D(depth_cnn[1])
        self.up1 = UpSample2D(depth_cnn[0])

        self.one_conv = Conv3D(1, (1,1,1), activation = 'sigmoid')

    def call(self, input_tensor):
        feats = self.feat_model(input_tensor)

        enc1_low = feats['enc1_low']
        enc2_low = feats['enc2_low']
        enc3_low = feats['enc3_low']

        x1 = self.enc1(input_tensor)
        x1_down = self.pool1(x1)
        x1_down = self.drop1(x1_down)

        x2 = concatenate([x1_down, enc1_low], axis = 4)
        x2 = self.enc2(x1_down)
        x2_down = self.pool2(x2)
        x2_down = self.drop2(x2_down)

        x3 = concatenate([x2_down, enc2_low], axis = 4)
        x3 = self.enc3(x2_down)
        x3_down = self.pool3(x3)
        x3_down = self.drop3(x3_down)

        base = concatenate([x3_down, enc3_low], axis = 4)
        base = self.bottleneck(x3_down)

        base_up = self.up3(base)
        c3 = concatenate([base_up, x3], axis = 4)
        d3 = self.dec3(c3)

        d3_up = self.up2(d3)
        c2 = concatenate([d3_up, x2], axis = 4)
        d2 = self.dec2(c2)

        d2_up = self.up1(d2)
        c1 = concatenate([d2_up, x1], axis = 4)
        d1 = self.dec1(c1)

        seg = self.one_conv(d1)
        return seg
    
    def build_graph(self, input_shape = (256, 256, 8, 1)):
        input = Input(shape = input_shape)
        return Model(inputs = [input], outputs = self.call(input))

class HybridRDUNet_002(Model):
    def __init__(self, base_model):
        super(HybridRDUNet_002, self).__init__()
        depth_cnn = [32, 64, 128, 256]

        self.feat_model = Extract2DFeat(base_model)

        self.enc1 = DenseConvBlock3D(depth_cnn[0])
        self.enc2 = DenseConvBlock3D(depth_cnn[1])
        self.enc3 = DenseConvBlock3D(depth_cnn[2])

        self.drop1 = SpatialDropout3D(0.1)
        self.drop2 = SpatialDropout3D(0.1)
        self.drop3 = SpatialDropout3D(0.1)

        self.pool1 = Pool2D()
        self.pool2 = Pool2D()
        self.pool3 = Pool2D()

        self.bottleneck = BNeckConvLSTM(depth_cnn[3])

        self.dec3 = DenseConvBlock3D(depth_cnn[2])
        self.dec2 = DenseConvBlock3D(depth_cnn[1])
        self.dec1 = DenseConvBlock3D(depth_cnn[0])

        self.up3 = UpSample2D(depth_cnn[2])
        self.up2 = UpSample2D(depth_cnn[1])
        self.up1 = UpSample2D(depth_cnn[0])

        self.one_conv = Conv3D(1, (1,1,1), activation = 'sigmoid')

    def call(self, input_tensor):
        feats = self.feat_model(input_tensor)

        enc1_low = feats['enc1_low']
        enc2_low = feats['enc2_low']
        enc3_low = feats['enc3_low']
        enc1_high = feats['enc1_high']
        enc2_high = feats['enc2_high']
        enc3_high = feats['enc3_high']

        x1 = self.enc1(input_tensor)
        x1_down = self.pool1(x1)
        x1_down = self.drop1(x1_down)

        x2 = concatenate([x1_down, enc1_low], axis = 4)
        x2 = self.enc2(x1_down)
        x2_down = self.pool2(x2)
        x2_down = self.drop2(x2_down)

        x3 = concatenate([x2_down, enc2_low], axis = 4)
        x3 = self.enc3(x2_down)
        x3_down = self.pool3(x3)
        x3_down = self.drop3(x3_down)

        base = concatenate([x3_down, enc3_low], axis = 4)
        base = self.bottleneck(x3_down)

        base_up = self.up3(base)
        c3 = concatenate([base_up, enc3_high], axis = 4)
        d3 = self.dec3(c3)

        d3_up = self.up2(d3)
        c2 = concatenate([d3_up, enc2_high], axis = 4)
        d2 = self.dec2(c2)

        d2_up = self.up1(d2)
        c1 = concatenate([d2_up, enc1_high], axis = 4)
        d1 = self.dec1(c1)

        seg = self.one_conv(d1)
        return seg
    
    def build_graph(self, input_shape = (256, 256, 8, 1)):
        input = Input(shape = input_shape)
        return Model(inputs = [input], outputs = self.call(input))

