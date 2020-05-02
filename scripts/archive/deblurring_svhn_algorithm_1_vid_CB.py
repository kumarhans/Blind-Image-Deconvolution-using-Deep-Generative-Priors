import tensorflow as tf
import keras.backend as K
from Utils import *
from generators.MotionBlurGenerator import *
from generators.SVHNGenerator import *
K.set_learning_phase(0)
from glob import glob
import os
import numpy as np
import ipdb
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

# paths
Orig_Path       = './results/hansSVHN/BlurSeq/*.png'
Blur_Path       = './results/hansSVHN/Original Blurs/Test Blurs.npy'

# paths
REGULARIZORS = [0.01 , 0.01]
RANDOM_RESTARTS = 10
NOISE_STD       = 0.01
STEPS           = 6000
IMAGE_RANGE = [0,1]
def step_size(t):
    return 0.01 * np.exp( - t / 1000 )

SAVE_PATH       = './results/hansSVHN/blurryNumberVid2/deblurring-alg1-'+str(int(NOISE_STD*100)) + 'percnoise-' +str(RANDOM_RESTARTS) + 'RR-VAE/deblurring_'
# -----------------------------------------------------------------------

# loading test blur kernels
W = np.load(Blur_Path) 
BLUR_RES = W.shape[1] #what is this?

# loading svhn test images
paths=[path for path in glob(Orig_Path)]
paths.sort()
Blurry_Images_orig = np.array([ imread(path) for path in paths]) / 255

Blurry_Images_orig = Blurry_Images_orig#[0:2]
print(Blurry_Images_orig.shape)
 
IMAGE_RES = Blurry_Images_orig.shape[1]
CHANNELS = Blurry_Images_orig.shape[-1]

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
Y_np=[]

#Converting to fourier form
Blurry_Images = []
for i in tqdm(range(len(Blurry_Images_orig)), ascii=True, desc ='Gen-Test-Blurry'):
    b_np = Blurry_Images_orig[i]
    y_np, y_f = toFourier(b_np, noise_std = NOISE_STD )
    Y_np.append(y_np)
    for _ in range(RANDOM_RESTARTS):
        Blurry_Images.append(y_f)

Y_np = np.array(Y_np)
Blurry_Images = np.array(Blurry_Images)

# Blurry_Images = np.array(Blurry_Images_orig)
# Blurry_Images,Blurry_Images_f= toFourier(Blurry_Image, NOISE_STD)
# alternating gradient descent for test images
image_gradients, blur_gradients, get_loss = Generate_Gradient_Functions(rr = Blurry_Images.shape[0],
                                                                        reg = REGULARIZORS, image_range = IMAGE_RANGE,
                                                                        decoder = svhn_decoder, blur_decoder = blur_decoder,
                                                                        image_res = IMAGE_RES, blur_res = BLUR_RES,
                                                                        channels = CHANNELS)

m_hat, h_hat, Loss = Optimize_Parallel(blurry_fourier = Blurry_Images, stepsize=step_size,steps = STEPS,
                                      image_grad = image_gradients , blur_grad = blur_gradients, 
                                      getloss = get_loss, latent_image_dim = svhn_latent_dim , latent_blur_dim = blur_latent_dim)
X_hat_test = []
W_hat_test = []
print(Loss.shape)
for i in range(len(Blurry_Images_orig)):
    m_hat_i = m_hat[i*RANDOM_RESTARTS:(i+1)*RANDOM_RESTARTS]
    h_hat_i = h_hat[i*RANDOM_RESTARTS:(i+1)*RANDOM_RESTARTS]
    Loss_i  =  Loss[i*RANDOM_RESTARTS:(i+1)*RANDOM_RESTARTS]
    x_hat_test, w_hat_test, loss_last_iter_test = Get_Min_Loss(Loss_i, m_hat_i, h_hat_i, decoder = svhn_decoder, blur_decoder = blur_decoder,
                                                               latent_image_dim = svhn_latent_dim, latent_blur_dim = blur_latent_dim,  print_grad=False)  
    X_hat_test.append(x_hat_test)
    W_hat_test.append(w_hat_test)

X_hat_test = np.array(X_hat_test)
W_hat_test = np.array(W_hat_test)

# saving results
Max = 10**len(str(len(Blurry_Images)-1))
print(Max)
for i in range(len(Blurry_Images_orig)):
    Save_Results(path = SAVE_PATH + str(i+Max)[1:], 
                     x_np = None, 
                     w_np = None,
                     y_np = Blurry_Images_orig[i], 
                     y_np_range = None,#Y_np_range[i] , 
                     x_hat_test = X_hat_test[i], 
                     w_hat_test = W_hat_test[i], 
                     x_range = None, 
                     x_hat_range = None,#X_hat_range[i], 
                     w_hat_range = None,#W_hat_range[i], 
                     clip=True)
K.get_session().close()
K.clear_session()

 



