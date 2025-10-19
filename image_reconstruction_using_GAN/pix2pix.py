###
### This file contains the Pix2Pix model, optimized for the purpose of image denoising 
### Largely based on https://www.tensorflow.org/tutorials/generative/pix2pix
### and https://opg.optica.org/boe/fulltext.cfm?uri=boe-12-10-6184&id=458664
### Last updated: 2022/05/04 9:15 AM
###

# Import required libraries
import tensorflow as tf
from matplotlib import pyplot as plt
from keras import layers, models
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Conv2DTranspose, Concatenate
from keras.optimizers import Adam

# Set the number of output channels. In this case, there is 
# only a single output channel (no RGB signal or whatsoever)
OUTPUT_CHANNELS = 1


### Helper functions
def downsample(filters, size, apply_batchnorm=True):
    """
    Define the downsampling steps that will be used in the discriminator and generator.
    Steps: Conv2D -> (BatchNorm) -> LeakyReLU
    """
    # Set random initialization
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    
    # Single convolutional layer
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))
    
    # Followed by BatchNorm (optional)
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    # Finally, a Leaky ReLU layer
    result.add(tf.keras.layers.LeakyReLU())
    return result


def upsample(filters, size, apply_dropout=False):
    """
    Define the downsampling steps that will be used in the discriminator and generator.
    Steps: Deconv2D -> BatchNorm -> (Dropout) -> ReLU
    """
    # Set random initialization
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    
    # Add single deconvolution layer 
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    # Followed by Batchnorm and dropout (optionally)
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    # Finally, a ReLU layer
    result.add(tf.keras.layers.ReLU())
    return result


# ### Define generator
# def Generator(input_size=(512,512,1), n_filters=64, kernel_size=4):
#     """
#     Define the generator of the Pix2Pix GAN model. Based on the U-Net model
#     """
#     # Define the inputs to the model
#     inputs = tf.keras.layers.Input(shape=input_size)

#     # Define the downwards (decoder) stream
#     down_stack = [
#         downsample(n_filters*1, kernel_size, apply_batchnorm=False),
#         downsample(n_filters*2, kernel_size), 
#         downsample(n_filters*4, kernel_size), 
#         downsample(n_filters*8, kernel_size),
#         downsample(n_filters*8, kernel_size),  
#         downsample(n_filters*8, kernel_size),  
#         #downsample(n_filters*8, kernel_size),  
#         #downsample(n_filters*8, kernel_size),  
#     ]

#     # Define the upwards (encoder) stream
#     up_stack = [
#         #upsample(n_filters*8, kernel_size, apply_dropout=True), 
#         #upsample(n_filters*8, kernel_size, apply_dropout=True), 
#         upsample(n_filters*8, kernel_size, apply_dropout=True), 
#         upsample(n_filters*8, kernel_size),
#         upsample(n_filters*4, kernel_size),
#         upsample(n_filters*2, kernel_size),
#         upsample(n_filters*1, kernel_size),
#     ]

#     # Set random initialization
#     initializer = tf.random_normal_initializer(0., 0.02)

#     # Define the final (deconvolution) layer (single output channel)
#     last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, kernel_size,
#                                             strides=2,
#                                             padding='same',
#                                             kernel_initializer=initializer,
#                                             activation='tanh')

#     x = inputs

#     # Downsampling the image through the model, holding onto the data 
#     # at different stages for the skip connections
#     skips = []
#     for down in down_stack:
#         x = down(x)
#         skips.append(x)

#     skips = reversed(skips[:-1])

#     # Upsampling the image through the model, concatenating with
#     # the 'skip'-connected images
#     for up, skip in zip(up_stack, skips):
#         x = up(x)
#         x = tf.keras.layers.Concatenate()([x, skip])

#     # Last layer
#     x = last(x)

#     # Define as a Keras model and return
#     return tf.keras.Model(inputs=inputs, outputs=x)

def residual_block(x,num_layer):
    # x = layers.Conv2D(num_layer, 1, activation=None,padding='same', kernel_initializer='glorot_uniform')(x)
    # x=layers.BatchNormalization()(x)
    # x=layers.ReLU()(x)
    # x = layers.Conv2D(num_layer, 3, strides=2,activation=None,padding='same', kernel_initializer='glorot_uniform')(x)
    # x=layers.BatchNormalization()(x)
    # x=layers.ReLU()(x)
    x=layers.MaxPooling2D(pool_size=(2, 2))(x)
    sh=x
    sh= layers.Conv2D(num_layer, 1, activation=None,padding='same', kernel_initializer='glorot_uniform')(sh)
    sh=layers.BatchNormalization()(sh)
    sh=layers.ReLU()(sh)
    conv_1 = layers.Conv2D(num_layer, 3, activation=None ,padding='same', kernel_initializer='glorot_uniform')(x)
    conv_1=layers.BatchNormalization()(conv_1)
    conv_1=layers.ReLU()(conv_1)
    conv_1 = layers.Conv2D(num_layer, 3, padding='same', kernel_initializer='glorot_uniform')(conv_1)
    conv_1=layers.BatchNormalization()(conv_1)
    
    conv_1+=sh
    conv_1=Activation('relu')(conv_1)
    return conv_1
def bn_relu(l):
    l=layers.BatchNormalization()(l)
    l=layers.ReLU()(l)
    return l

def gen(input_size):
    inputs = Input(shape=input_size)
    sh1=layers.Conv2D(64, 1, activation=None,padding='same', kernel_initializer='glorot_uniform')(inputs)
    sh1=bn_relu(sh1)
    conv1 = layers.Conv2D(32, 3, activation=None ,padding='same', kernel_initializer='glorot_uniform')(inputs)
    conv1=bn_relu(conv1)
    conv1 = layers.Conv2D(64, 3, activation=None, padding='same', kernel_initializer='glorot_uniform')(conv1)
    conv1=bn_relu(conv1)    
    conv1+=sh1

    conv2=residual_block(conv1,128)
    conv3=residual_block(conv2,256)
    conv4=residual_block(conv3,512)
    sh2=conv4
    # conv4 = layers.Conv2D(512, 1, activation=None,padding='same', kernel_initializer='glorot_uniform')(conv4)
    # conv4=bn_relu(conv4)
    # conv4 = layers.Conv2D(512, 3, strides=2,activation=None,padding='same', kernel_initializer='glorot_uniform')(conv4)
    # conv4=bn_relu(conv4)
    conv4=layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = layers.Conv2D(1024, 3, activation=None, padding='same', kernel_initializer='glorot_uniform')(conv4)
    conv5=bn_relu(conv5)

    up1=layers.Conv2DTranspose(512, 3, strides=2,activation=None, padding='same',kernel_initializer='glorot_uniform')(conv5)
    up1=bn_relu(up1)
    merge1 = layers.concatenate([up1, sh2], axis=-1)
    conv7 = layers.Conv2D(256, 1)(merge1)
    conv7 = layers.Conv2D(512, 3, activation=None, padding='same', kernel_initializer='glorot_uniform')(conv7)
    conv7=bn_relu(conv7)


    up2=layers.Conv2DTranspose(256, 3, strides=2,activation=None, padding='same',kernel_initializer='glorot_uniform')(conv7)
    up2=bn_relu(up2)
    merge2 = layers.concatenate([up2, conv3], axis=-1)
    conv8 = layers.Conv2D(128, 1)(merge2)
    conv8 = layers.Conv2D(256, 3, activation=None, padding='same', kernel_initializer='glorot_uniform')(conv8)
    conv8=bn_relu(conv8)

    up3=layers.Conv2DTranspose(128, 3, strides=2,activation=None, padding='same',kernel_initializer='glorot_uniform')(conv8)
    up3=bn_relu(up3)
    merge3 = layers.concatenate([up3, conv2], axis=-1)
    conv9 = layers.Conv2D(64, 1)(merge3)
    conv9 = layers.Conv2D(128, 3, activation=None, padding='same', kernel_initializer='glorot_uniform')(conv9)
    conv9=bn_relu(conv9)

    up4=layers.Conv2DTranspose(64, 3, strides=2,activation=None, padding='same',kernel_initializer='glorot_uniform')(conv9)
    up4=bn_relu(up4)
    merge4 = layers.concatenate([up4, conv1], axis=-1)
    conv10 = layers.Conv2D(32, 1)(merge4)
    conv10 = layers.Conv2D(64, 3, activation=None ,padding='same', kernel_initializer='glorot_uniform')(conv10)
    conv10=bn_relu(conv10)
    conv10 = layers.Conv2D(1, 1)(conv10)
    output = conv10+inputs
    
    model = models.Model(inputs=inputs, outputs=output)
    # optimizer = Adam(learning_rate=learning_rate)
    # model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

### Define generator loss
# def generator_loss(disc_generated_output, gen_output, target, lambda_value=1,loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)):
def generator_loss(disc_generated_output,gen_output, target,loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)):
    
    """
    Define the generator loss. Contains both the sigmoid cross-entropy loss of the 
    generated images compared to an array of ones, and the L1 loss between generated and target images.
    
    Total generator loss = GAN loss + LAMBDA * L1-loss.

    A value of 100 for LAMBDA was found by the authors of the Pix2Pix paper.
    """
    # gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.square(target - gen_output)) # Mean absolute error
    # total_gen_loss = gan_loss + (100 * l1_loss)
    # return total_gen_loss, gan_loss, l1_loss
    return l1_loss


# ### Define discriminator
# def Discriminator(input_shape=(512,512,1)):
#     """
#     Define the discriminator of the Pix2Pix GAN model.
#     Structure:
    
#     Conv2D (64; 4x4)  --> LeakyReLU
#      v
#     Conv2D (128; 4x4) --> BatchNorm --> LeakyReLU
#      v
#     Conv2D (256; 4x4) --> BatchNorm --> LeakyReLU
#      v
#     Zero padding
#      V 
#     Conv2D (512, 4, stride=1) --> BatchNorm --> LeakyReLU
#      V
#     Zero padding --> Conv2D (1, 4x4, stride=1)
#     """
#     # Set random initialization
#     initializer = tf.random_normal_initializer(0., 0.02)

#     # Define two types of input: the actual input image and the 'target' image.
#     # The target image is either the reference image or the predicted image
#     inp = tf.keras.layers.Input(shape=input_shape, name='input_image')
#     tar = tf.keras.layers.Input(shape=input_shape, name='target_image')

#     # Concatenate the inputs
#     x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

#     # Perform three downsampling steps
#     down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
#     down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
#     down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

#     # Followed by zero padding, and another set of Conv2D -> BatchNorm -> LeakyReLU
#     # Difference with downsampling is that now, a stride of 1 is used instead of 2
#     zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
#     conv = tf.keras.layers.Conv2D(512, 4, strides=1,
#                                     kernel_initializer=initializer,
#                                     use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

#     batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
#     leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

#     # Do another zero padding and Conv2D, end up with a single channel
#     zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)
#     last = tf.keras.layers.Conv2D(1, 4, strides=1,
#                                     kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

#     # Define as a Keras model and return
#     return tf.keras.Model(inputs=[inp, tar], outputs=last)


# def Discriminator(input_shape=(512,512,1)):
#     # initializer = tf.random_normal_initializer(0., 0.02)
#     inp = tf.keras.layers.Input(shape=input_shape, name='input_image')
#     tar = tf.keras.layers.Input(shape=input_shape, name='target_image')
#     # inputs1 = Input(shape=input_size1)
#     x = tf.keras.layers.concatenate([inp, tar])
#     conv1=layers.Conv2D(64, kernel_size=3, padding='same', input_shape=(None, None, 1))(x)
#     conv1=layers.LeakyReLU(0.2)(conv1)

#     conv2=layers.Conv2D(64, kernel_size=3, strides=2, padding='same')(conv1)
#     conv2=layers.BatchNormalization()(conv2)
#     conv2=layers.LeakyReLU(0.2)(conv2)

#     conv2=layers.Conv2D(128, kernel_size=3, strides=2, padding='same')(conv2)
#     conv2=layers.BatchNormalization()(conv2)
#     conv2=layers.LeakyReLU(0.2)(conv2)

#     conv2=layers.Conv2D(128, kernel_size=3, strides=2, padding='same')(conv2)
#     conv2=layers.BatchNormalization()(conv2)
#     conv2=layers.LeakyReLU(0.2)(conv2)

#     conv2=layers.Conv2D(256, kernel_size=3, strides=2, padding='same')(conv2)
#     conv2=layers.BatchNormalization()(conv2)
#     conv2=layers.LeakyReLU(0.2)(conv2)

#     conv2=layers.Conv2D(256, kernel_size=3, strides=2, padding='same')(conv2)
#     conv2=layers.BatchNormalization()(conv2)
#     conv2=layers.LeakyReLU(0.2)(conv2)

#     conv2=layers.Conv2D(512, kernel_size=3, strides=2, padding='same')(conv2)
#     conv2=layers.BatchNormalization()(conv2)
#     conv2=layers.LeakyReLU(0.2)(conv2)
    

    # conv2=layers.Conv2D(512, kernel_size=3, strides=2, padding='same')(conv2)
    # conv2=layers.BatchNormalization()(conv2)
    # conv2=layers.LeakyReLU(0.2)(conv2)

    # # conv2=layers.GlobalAveragePooling2D()(conv2)
    # flat = layers.Flatten()(conv2)
    # # dense1=layers.Dense(16)(conv2)
    # dense1=layers.LeakyReLU(0.2)(flat)
    # last = tf.keras.layers.Dense(1, activation='sigmoid')(dense1)
    # model = models.Model(inputs=[inp, tar], outputs=last)
    # return model

def discriminator(input_shape):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, kernel_size=3, padding='same')(inp)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(64, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(128, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(256, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(512, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(512, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(0.2)(x)
    output = layers.Dense(1)(x)

    model = models.Model(inputs=inp, outputs=output)
    return model

### Define discriminator loss
def discriminator_loss(disc_real_output, disc_generated_output,\
                       loss_object= tf.keras.losses.BinaryCrossentropy(from_logits=True)):
    """
    Define the discriminator loss. Requires the real and generated images.
    
    Real loss: sigmoid cross-entropy loss of the real images compared to an array of ones ('real' images)
    Generated loss: sigmoid cross-entropy loss of the generated images and an array of zeros ('fake' images)
    Total discriminator loss = real loss + generated loss
    """

    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


def grad_penalty(D, xr, xf):
    """
    Gradient penalty for Discriminator of Wasserstein GAN
    D: Discriminator model, xr: (N, H, W, C), xf: (N, H, W, C)
    """

    # Generate random values for mixing
    t = tf.random.uniform(shape=[xr.shape[0], 1, 1, 1], minval=0.0, maxval=1.0)
    xm = t * xr + (1 - t) * xf
    xm = tf.Variable(xm, trainable=True, dtype=tf.float32)

    # Forward pass through the discriminator
    with tf.GradientTape() as tape:
        WDmid = D(xm)

    # Compute the gradient of the output with respect to the input
    Gradmid = tape.gradient(WDmid, xm)
    Gradmid = tf.reshape(Gradmid, (tf.shape(Gradmid)[0], -1))

    # Compute the gradient penalty
    GP = tf.reduce_mean(tf.square(tf.norm(Gradmid, axis=1) - 1))

    return GP

def ssim_loss(input, target):
    """
    Inputs:
    - input: TensorFlow Tensor of shape (N, H, W, C).
    - target: TensorFlow Tensor of shape (N, H, W, C).

    Returns:
    - mean SSIM
    """
    ssim = tf.image.ssim(input, target, max_val=1.0)
    return -tf.math.log(tf.reduce_mean(ssim))

# def GradPenalty(D, xr, xf):
#     """
#     Gradient penalty for Discriminator of  Wasserstein GAN
#     D: Discriminator model, xr: (N,C,H,W), xf:(N,C,H,W)
#     torch.autograd.grad(), refer to:
#     # https://blog.csdn.net/sinat_28731575/article/details/90342082
#     # https://zhuanlan.zhihu.com/p/33378444
#     # https://zhuanlan.zhihu.com/p/29923090
#     """

#     t = torch.randn(xr.size(0), 1, 1, 1).type(datype)
#     xm = t*xr.clone() + (1-t)*xf.clone()
#     xm.requires_grad_(True)
#     WDmid = D(xm)
#     # Compute the gradients of outputs w.r.t. the inputs.(same size as inputs)
#     # grad_outputs: The “vector” in the Jacobian-vector product, usually all one. (same size as outputs)
#     # create_graph: to equip GP with a grad_fn.
#     # retain_graph: retain the graph used to compute the grad for the backward of GP
#     Gradmid = torch.autograd.grad(outputs=WDmid, inputs=xm,
#                                   grad_outputs=torch.ones_like(
#                                       WDmid).type(datype),
#                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
#     Gradmid = Gradmid.view(Gradmid.size(0), -1)
#     GP = torch.pow((Gradmid.norm(2, dim=1)-1), 2).mean()
#     return GP