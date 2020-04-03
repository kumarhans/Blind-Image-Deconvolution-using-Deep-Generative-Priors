import tensorflow as tf
import keras.backend as K
from Utils import *
from generators.MotionBlurGenerator import *
from generators.SVHNGenerator import *
K.set_learning_phase(0)
from glob import glob
import os
import numpy as np


# paths
Orig_Path       = './results/SVHN/blurryNumberVid/*.jpg'
Blur_Path       = './results/SVHN/Original Blurs/Test Blurs.npy'

# paths
REGULARIZORS = [0.01 , 0.01]
RANDOM_RESTARTS = 3
NOISE_STD       = 0.01
STEPS           = 6000
IMAGE_RANGE = [0,1]
def step_size(t):
    return 0.01 * np.exp( - t / 1000 )

SAVE_PATH       = './results/SVHN/blurryNumberVid/deblurring - alg1 - '+str(int(NOISE_STD*100)) + 'perc noise - ' +str(RANDOM_RESTARTS) + 'RR-VAE/deblurring_'
# -----------------------------------------------------------------------

# loading test blur kernels
W = np.load(Blur_Path) 
BLUR_RES = W.shape[1]

# loading svhn test images
X_Orig = np.array([ imread(path) for path in glob(Orig_Path)]) / 255
X_Orig = X_Orig[0:1]
print(X_Orig.shape)
 
IMAGE_RES = X_Orig.shape[1]
CHANNELS = X_Orig.shape[-1]

# loading svhn generator
SVHNGen = SVHNGenerator()
SVHNGen.GenerateModel()
SVHNGen.LoadWeights()
svhn_vae, svhn_encoder, svhn_decoder = SVHNGen.GetModels()
svhn_latent_dim = SVHNGen.latent_dim

# loading motion blur generator
BLURGen = MotionBlur()
BLURGen.GenerateModel()
BLURGen.LoadWeights()
blur_vae, blur_encoder, blur_decoder = BLURGen.GetModels()
blur_latent_dim = BLURGen.latent_dim


# W = np.load(Blur_Path) 
# BLUR_RES = W.shape[1]


# check if save dir exists, if not create a new one
try:
    os.stat(SAVE_PATH[:-12])
except:
    os.mkdir(SAVE_PATH[:-12])

# generating blurry images from test
Y_np = []
 
 
Blurry_Images_range = np.array(X_Orig)

 

# alternating gradient descent for test images
image_gradients, blur_gradients, get_loss = Generate_Gradient_Functions(rr = Blurry_Images_range.shape[0],
                                                                        reg = REGULARIZORS, image_range = IMAGE_RANGE,
                                                                        decoder = svhn_decoder, blur_decoder = blur_decoder,
                                                                        image_res = IMAGE_RES, blur_res = BLUR_RES,
                                                                        channels = CHANNELS)

# alternating gradient descent for range images 
m_hat, h_hat, Loss = Optimize_Parallel(blurry_fourier = Blurry_Images_range, stepsize=step_size,steps = STEPS,
                                      image_grad = image_gradients , blur_grad = blur_gradients, 
                                      getloss = get_loss, latent_image_dim = svhn_latent_dim , latent_blur_dim = blur_latent_dim)
X_hat_range = []
W_hat_range = []
for i in range(len(X_Orig)):
    m_hat_i = m_hat[i*RANDOM_RESTARTS:(i+1)*RANDOM_RESTARTS]
    h_hat_i = h_hat[i*RANDOM_RESTARTS:(i+1)*RANDOM_RESTARTS]
    print(i*RANDOM_RESTARTS)
    print(Loss[i*RANDOM_RESTARTS:(i+1)*RANDOM_RESTARTS])
    Loss_i  = Loss[i*RANDOM_RESTARTS:(i+1)*RANDOM_RESTARTS]
    x_hat_range, w_hat_range, loss_last_iter_range = Get_Min_Loss(Loss_i, m_hat_i, h_hat_i, decoder = svhn_decoder, blur_decoder = blur_decoder,
                                                                  latent_image_dim = svhn_latent_dim, latent_blur_dim = blur_latent_dim, print_grad=False)  
    X_hat_range.append(x_hat_range)
    W_hat_range.append(w_hat_range)

X_hat_range = np.array(X_hat_range)
W_hat_range = np.array(W_hat_range)





# saving results
Max = 10**len(str(len(X_Orig)-1))
for i in range(len(X_Orig)):
    Save_Results(path = SAVE_PATH + str(i+Max)[1:], 
                     x_np = None, 
                     w_np = None,
                     y_np = None, 
                     y_np_range = None, 
                     x_hat_test = None, 
                     w_hat_test = None,
                     x_range = None, 
                     x_hat_range = X_hat_range[i], 
                     w_hat_range = W_hat_range[i], 
                     clip=True)

 



