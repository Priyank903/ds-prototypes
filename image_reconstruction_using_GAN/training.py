###
### This file contains the functions that are required for training the
### Pix2Pix GAN model. For that reason, it uses the functions from 
### pix2pix.py.
### Last updated: 2022/05/04 9:15 AM
###

# Import libraries
import tensorflow as tf
import os
import numpy as np
from datetime import datetime
from time import time
from pix2pix import gen, discriminator, generator_loss, discriminator_loss,grad_penalty,ssim_loss
from matplotlib import pyplot as plt
from tqdm import tqdm

from keras import backend as K_backend
K_backend.set_image_data_format('channels_last')
tf.config.run_functions_eagerly(True)

# Set constants
CHECKPOINT_PATH = './training_checkpoints/'
LOG_PATH        = './logs/'

# Define TensorBoard writer
SUMMARY_WRITER = tf.summary.create_file_writer(
  os.path.join(LOG_PATH, "tb", datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))
)

# Define batch generator (used in training loop)
def batch_generator(x, n):
    """
    X: data
    n: batch size
    """
    start, stop = 0, n
    while True:
        if start < stop:
            yield x[start:stop]
        else:
            break
        start = stop
        stop = (stop + n) % len(x)


### Do a single training step
@tf.function
def train_step(input_image, target, step, generator, discriminator, gen_optim, disc_optim, iter):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Let the model generate an image based on an input image, let it train as well
        gen_output = generator(input_image, training=True)
        print(step,iter)
        # if  step==99:
        #     # plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
        #     # plt.imshow(target[0],cmap='gray')  # Use an appropriate colormap
        #     # plt.title(f'GT')
        #     # plt.savefig(f'/mnt/sdb1/priyank/project/GT_80_train/GT_{step}_{iter}.png', format='png', bbox_inches='tight')
        #     # plt.show()
            
        #     plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
        #     plt.imshow(gen_output[0],cmap='gray')  # Use an appropriate colormap
        #     plt.title(f'generator_prediction')
        #     plt.savefig(f'/mnt/sdb1/priyank/project/gen_pred_70_train_new_14jan/pred_{step}_{iter}.png', format='png', bbox_inches='tight')
        #     plt.show() 

            # plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
            # plt.imshow(input_image[0],cmap='gray')  # Use an appropriate colormap
            # plt.title(f'artiactual input')
            # plt.savefig(f'/mnt/sdb1/priyank/project/arti_80_train/arti_{step}_{iter}.png', format='png', bbox_inches='tight')
            # plt.show()  



        # Let the discriminator learn, first input an input image and the reference image ('real'), 
        # then the input image and the generated output ('fake')
        disc_real_output = discriminator(target, training=True)
        disc_generated_output = discriminator(gen_output, training=True)

        GP = 10 * grad_penalty(discriminator, target, gen_output)

        # # Wasserstein GAN Loss
        lossD = disc_generated_output.mean() - disc_real_output.mean() + GP
        print(disc_real_output,disc_generated_output,GP,lossD)        
        # (x, y) = self.data_iter.__next__()
        # AR_img, OR_img = x.to(DEVICE), y.to(DEVICE)
        # AR_img.requires_grad_(True)
        # OR_img.requires_grad_(True)

        # SR_img = self.G(AR_img)
        # D_fake = self.D(SR_img)

        # # LossG
        # Gloss = Gloss_fn(self.loss_type, SR_img, OR_img)
        Lloss = generator_loss(gen_output, target)
        Adloss = ssim_loss(gen_output, target)
        Gloss = Lloss + 2e-2*Adloss
        lossG_adv = -1e-3 * disc_generated_output.mean()
        lossG = Gloss[-1] + lossG_adv
        
        

        # Calculate the losses for both models 
        # gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        # gen_total_loss = generator_loss(gen_output, target, lambda_value=L1_lambda)
        # disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
        # print(disc_loss.numpy(),disc_real_output.numpy(), disc_generated_output.numpy())

    # Calculate the gradients for both models based on their losses
    generator_gradients = gen_tape.gradient(lossG,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(lossD,
                                                discriminator.trainable_variables)

    # Optimize the models by applying the calculated gradients
    gen_optim.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    disc_optim.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    # Write the losses to TensorBoard
    # with SUMMARY_WRITER.as_default():
    #     tf.summary.scalar('gen_total_loss', gen_total_loss, step=step)
    #     tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step)
    #     tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step)
    #     tf.summary.scalar('disc_loss', disc_loss, step=step)
    
    # return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss

# @tf.function
# def train_step_gen(input_image, target, step, generator, discriminator, gen_optim, disc_optim, iter):
#     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#         # Let the model generate an image based on an input image, let it train as well
#         gen_output = generator(input_image, training=True)
#         print(step,iter)
#         # if  (step%10==0  and step!=0) or step==99:
#         #     # plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
#         #     # plt.imshow(target[0],cmap='gray')  # Use an appropriate colormap
#         #     # plt.title(f'GT')
#         #     # plt.savefig(f'/mnt/sdb1/priyank/project/GT_80_train/GT_{step}_{iter}.png', format='png', bbox_inches='tight')
#         #     # plt.show()
            
#         #     plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
#         #     plt.imshow(gen_output[0],cmap='gray')  # Use an appropriate colormap
#         #     plt.title(f'generator_prediction')
#         #     plt.savefig(f'/mnt/sdb1/priyank/project/gen_pred_70_train_new_15jan/pred_{iter}.png', format='png', bbox_inches='tight')
#         #     plt.show() 

#             # plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
#             # plt.imshow(input_image[0],cmap='gray')  # Use an appropriate colormap
#             # plt.title(f'artiactual input')
#             # plt.savefig(f'/mnt/sdb1/priyank/project/arti_80_train/arti_{step}_{iter}.png', format='png', bbox_inches='tight')
#             # plt.show()  



#         # Let the discriminator learn, first input an input image and the reference image ('real'), 
#         # then the input image and the generated output ('fake')
#         disc_real_output = discriminator([input_image, target], training=True)
#         disc_generated_output = discriminator([input_image, gen_output], training=True)

#         # Calculate the losses for both models 
#         gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
#         # gen_total_loss = generator_loss(gen_output, target, lambda_value=L1_lambda)
#         # disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
#         # print(disc_loss.numpy(),disc_real_output.numpy(), disc_generated_output.numpy())

#     # Calculate the gradients for both models based on their losses
#     generator_gradients = gen_tape.gradient(gen_total_loss,
#                                             generator.trainable_variables)
#     # discriminator_gradients = disc_tape.gradient(disc_loss,
#     #                                             discriminator.trainable_variables)

#     # Optimize the models by applying the calculated gradients
#     gen_optim.apply_gradients(zip(generator_gradients,
#                                             generator.trainable_variables))
#     # disc_optim.apply_gradients(zip(discriminator_gradients,
#     #                                             discriminator.trainable_variables))



### Main training function
def model_train(X_tr, y_tr, X_val,y_val,epochs, img_shape=(512,512,1), n_layers=64, lr=2e-4, \
                 batch_size=1):
    
    # Initialize models
    G = gen(img_shape)
    D = discriminator(img_shape)

    # Initialize Adam optimizers
    gen_optim = tf.keras.optimizers.Adam(lr, beta_1=0.5)
    disc_optim = tf.keras.optimizers.Adam(lr, beta_1=0.5)

    # Set up checkpoints for saving the model
    checkpoint_prefix = os.path.join(CHECKPOINT_PATH, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optim,
                                    discriminator_optimizer=disc_optim,
                                    generator=G,
                                    discriminator=D)
    
    # Set up dictionary to save the losses
    global losses_dict 
    losses_dict = {'gen_total_loss': [], \
                   'gen_gan_loss'  : [], \
                   'gen_l1_loss'   : [], \
                   'disc_loss'     : []}

    # Start the loop for X epochs
    for epoch in range(epochs):
        print(f"Epoch {epoch}: ", end='')
        # Set a timer
        # start = time()
        iter=0
        iter_test=0

        # initiate batch generator
        batches_X_train = batch_generator(X_tr, batch_size)
        batches_y_train = batch_generator(y_tr, batch_size)
        batches_X_test = batch_generator(X_val, batch_size)
        batches_y_test = batch_generator(y_val, batch_size)


        for image_batch, ref_batch in tqdm(zip(batches_X_train, batches_y_train)):
            # Change data types for compatibility
            image_batch = image_batch.astype('float32')
            ref_batch   = ref_batch.astype('float32')
            image_batch = tf.reshape(image_batch, (1, 512, 512, 1))
            ref_batch = tf.reshape(ref_batch,(1, 512, 512, 1))
            # if epoch==99:
            #     plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
            #     plt.imshow(ref_batch[0],cmap='gray')  # Use an appropriate colormap
            #     plt.title(f'GT')
            #     plt.savefig(f'/mnt/sdb1/priyank/project/GT_70_train_new_15jan/GT_{epoch}_{iter}.png', format='png', bbox_inches='tight')
            #     plt.show()
            #     plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
            #     plt.imshow(image_batch[0],cmap='gray')  # Use an appropriate colormap
            #     plt.title(f'artiactual input')
            #     plt.savefig(f'/mnt/sdb1/priyank/project/arti_70_train_new_15jan/arti_{epoch}_{iter}.png', format='png', bbox_inches='tight')
            #     plt.show()
            # if iter>55:
            # gen_tot, gen_gan, gen_l1, disc = train_step(image_batch, image_batch-ref_batch, epoch, G, D, gen_optim, disc_optim,iter, L1_lambda)
            # train_step_gen(image_batch, ref_batch, epoch, G, D, gen_optim, disc_optim,iter)
            # if iter%3==0 and iter!=0:
            train_step(image_batch, ref_batch, epoch, G, D, gen_optim, disc_optim,iter)
            iter+=1
        # if epoch==99:
        #     for image_batch_test, ref_batch_test in tqdm(zip(batches_X_test, batches_y_test)):
        #         # Change data types for compatibility
        #         image_batch_test = image_batch_test.astype('float32')
        #         ref_batch_test   = ref_batch_test.astype('float32')
        #         image_batch_test = tf.reshape(image_batch_test, (1, 512, 512, 1))
        #         ref_batch_test = tf.reshape(ref_batch_test,(1, 512, 512, 1))

        #         plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
        #         plt.imshow(ref_batch_test[0],cmap='gray')  # Use an appropriate colormap
        #         plt.title(f'GT')
        #         plt.savefig(f'/mnt/sdb1/priyank/project/GT_30_test_new_15jan/GT_{epoch}_{iter_test}.png', format='png', bbox_inches='tight')
        #         plt.show()
                
        #         plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
        #         plt.imshow(image_batch_test[0],cmap='gray')  # Use an appropriate colormap
        #         plt.title(f'artiactual input')
        #         plt.savefig(f'/mnt/sdb1/priyank/project/arti_30_test_new_15jan/arti_{epoch}_{iter_test}.png', format='png', bbox_inches='tight')
        #         plt.show()
        #         pred=G.predict(image_batch_test)
        #         plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
        #         plt.imshow(pred[0],cmap='gray')  # Use an appropriate colormap
        #         plt.title(f'gen_pred')
        #         plt.savefig(f'/mnt/sdb1/priyank/project/gen_disc_pred_30_new_15jan/pred_{epoch}_{iter_test}.png', format='png', bbox_inches='tight')
        #         plt.show()
        #         iter_test+=1







        # # Write the losses from the end of the epoch to a dictionary
        # losses_dict['gen_total_loss'].append(gen_tot.numpy())
        # losses_dict['gen_gan_loss'].append(gen_gan.numpy())
        # losses_dict['gen_l1_loss'].append(gen_l1.numpy())
        # losses_dict['disc_loss'].append(disc.numpy())

        # # Save the model every 15 epochs
        # if (epoch + 1) % 15 == 0:
        #     checkpoint.save(file_prefix = checkpoint_prefix)

        # print('Time for epoch {} is {} sec'.format(epoch + 1, time()-start))

    return G, D

