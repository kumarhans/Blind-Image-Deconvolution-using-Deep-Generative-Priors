import tensorflow as tf
import keras.backend as K
from Utils import *
from generators.MotionBlurGenerator import *
# from generators.SVHNGenerator import * 
from generators.SVHN_New import *
from generators.SVHNgan import SVHNganGenerator
K.set_learning_phase(0)
from glob import glob
import os
import numpy as np
from PIL import Image
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

# paths
Orig_Path       = './results/SVHN/Original Images/*.png'
Range_Path      = './results/SVHN/Range Images/*.png'
Blur_Path       = './results/SVHN/Original Blurs/Test Blurs.npy'

# paths
REGULARIZORS = [0.01 , 0.01]
RANDOM_RESTARTS = 10
NOISE_STD       = 0.01
STEPS           = 6000
LEARNING_RATE = 0.005
IMAGE_RANGE = [0,1]
optimizer       = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
def step_size(t):
    return 0.01 * np.exp( - t / 1000 )

SAVE_PATH       = './results/SVHN/deblur - '+str(int(NOISE_STD*100)) + 'perc noise - ' +str(RANDOM_RESTARTS) + 'RR/out_'
# -----------------------------------------------------------------------

# loading test blur kernels
W = np.load(Blur_Path) 
BLUR_RES = W.shape[1]

# loading svhn test images
X_Orig = []
for filename in glob(Orig_Path):
    im = Image.open(filename)
    im = np.array(im)
    X_Orig.append(im)
X_Orig = np.array(X_Orig) / 255
# print(X_Orig.shape)
X_Orig = X_Orig[0:10]
# print(X_Orig.shape)
IMAGE_RES = X_Orig.shape[1]
CHANNELS = X_Orig.shape[-1]

# loading new generator model
SVHNGen_new = SVHN_New()
SVHNGen_new.GenerateModel()
SVHNGen_new.LoadWeights()
svhn_vae_new, svhn_encoder_new, svhn_decoder_new = SVHNGen_new.GetModels()
svhn_latent_dim_new = SVHNGen_new.latent_dim

# loading svhn generator
SVHNGen = SVHNganGenerator()
SVHNGen.GenerateModel()
SVHNGen.LoadWeights()
SVHNGAN = SVHNGen.GetModels()
svhn_latent_dim = SVHNGen.latent_dim

# loading motion blur generator
BLURGen = MotionBlur()
BLURGen.GenerateModel()
BLURGen.LoadWeights()
blur_vae, blur_encoder, blur_decoder = BLURGen.GetModels()
blur_latent_dim = BLURGen.latent_dim

# check if save dir exists, if not create a new one
try:
    os.stat(SAVE_PATH[:-5])
except:
    os.mkdir(SAVE_PATH[:-5])

# generating blurry images from test
Y_np = []
Blurry_Images = []
for i in tqdm(range(len(X_Orig)), ascii=True, desc ='Gen-Test-Blurry'):
    x_np = X_Orig[i]
    w_np = W[i]
    y_np, y_f = GenerateBlurry(x_np, w_np, noise_std = NOISE_STD )
    Y_np.append(y_np)
    for _ in range(RANDOM_RESTARTS):
        Blurry_Images.append(y_f)

Y_np = np.array(Y_np)
Blurry_Images = np.array(Blurry_Images)


# alternating gradient descent for test images using new weights
image_gradients, blur_gradients, get_loss = Generate_Gradient_Functions(rr = Blurry_Images.shape[0],
                                                                        reg = REGULARIZORS, image_range = IMAGE_RANGE,
                                                                        decoder = svhn_decoder_new, blur_decoder = blur_decoder,
                                                                        image_res = IMAGE_RES, blur_res = BLUR_RES,
                                                                        channels = CHANNELS)
m_hat, h_hat, Loss = Optimize_Parallel(blurry_fourier = Blurry_Images, stepsize=step_size,steps = STEPS,
                                      image_grad = image_gradients , blur_grad = blur_gradients, 
                                      getloss = get_loss, latent_image_dim = svhn_latent_dim , latent_blur_dim = blur_latent_dim)


X_hat_new = []
W_hat_new = []
for i in range(len(X_Orig)):
    m_hat_i = m_hat[i*RANDOM_RESTARTS:(i+1)*RANDOM_RESTARTS]
    h_hat_i = h_hat[i*RANDOM_RESTARTS:(i+1)*RANDOM_RESTARTS]
    Loss_i  =  Loss[i*RANDOM_RESTARTS:(i+1)*RANDOM_RESTARTS]
    x_hat_test, w_hat_test, loss_last_iter_test = Get_Min_Loss(Loss_i, m_hat_i, h_hat_i, decoder = svhn_decoder_new, blur_decoder = blur_decoder,
                                                               latent_image_dim = svhn_latent_dim, latent_blur_dim = blur_latent_dim,  print_grad=False)  
    X_hat_new.append(x_hat_test)
    W_hat_new.append(w_hat_test)

X_hat_new = np.array(X_hat_new)
W_hat_new = np.array(W_hat_new)

# saving results
Max = 10**len(str(len(X_Orig)-1))
for i in range(len(X_Orig)):
    Save_Results(path = SAVE_PATH + str(i+Max)[1:], 
                     x_np = X_Orig[i], 
                     w_np = W[i],
                     y_np = Y_np[i], 
                     y_np_range = None , 
                     x_hat_test = None, 
                     w_hat_test = None, 
                     x_range = None, 
                     x_hat_range = X_hat_new[i], 
                     w_hat_range = W_hat_new[i], 
                     clip=True)


k = 0

# Blurred
for i in range(len(X_Orig)):
    plt.subplot(4,len(X_Orig),k+1)
    plt.imshow(Y_np[i])
    plt.axis("Off")
    k=k+1

# VAE Result
for i in range(len(X_Orig)):
    plt.subplot(4,len(X_Orig),k+1)
    plt.imshow(X_hat_new[i])
    plt.axis("Off")
    k=k+1

# Running GAN

# Algo 1
# constants
REGULARIZORS = [0.01 , 0.01]
RANDOM_RESTARTS = 10
NOISE_STD       = 0.01
STEPS           = 10000
IMAGE_RANGE = [-1,1]


# loading svhn generator
SVHNGen = SVHNganGenerator()
SVHNGen.GenerateModel()
SVHNGen.LoadWeights()
SVHNGAN = SVHNGen.GetModels()
svhn_latent_dim = SVHNGen.latent_dim

# alternating gradient descent for test images
image_gradients, blur_gradients, get_loss = Generate_Gradient_Functions(rr = Blurry_Images.shape[0],
                                                                        reg = REGULARIZORS, image_range = IMAGE_RANGE,
                                                                        decoder = SVHNGAN , blur_decoder = blur_decoder,
                                                                        image_res = IMAGE_RES, blur_res = BLUR_RES,
                                                                        channels = CHANNELS)
m_hat, h_hat, Loss = Optimize_Parallel(blurry_fourier = Blurry_Images, stepsize=step_size,steps = STEPS,
                                      image_grad = image_gradients , blur_grad = blur_gradients, 
                                      getloss = get_loss, latent_image_dim = svhn_latent_dim , latent_blur_dim = blur_latent_dim)
X_hat_test = []
W_hat_test = []
for i in range(len(X_Orig)):
    m_hat_i = m_hat[i*RANDOM_RESTARTS:(i+1)*RANDOM_RESTARTS]
    h_hat_i = h_hat[i*RANDOM_RESTARTS:(i+1)*RANDOM_RESTARTS]
    Loss_i  = Loss[i*RANDOM_RESTARTS:(i+1)*RANDOM_RESTARTS]
    x_hat_test, w_hat_test, loss_last_iter_test = Get_Min_Loss(Loss_i, m_hat_i, h_hat_i, decoder = SVHNGAN , blur_decoder = blur_decoder,
                                                               latent_image_dim = svhn_latent_dim, latent_blur_dim = blur_latent_dim,  print_grad=False)  
    X_hat_test.append(x_hat_test)
    W_hat_test.append(w_hat_test)

X_hat_test = np.array(X_hat_test)
W_hat_test = np.array(W_hat_test)

X_hat_test = (X_hat_test + 1)/2
Max = 10**len(str(len(X_Orig)-1))

# saving results
for i in range(len(X_Orig)):
    Save_Results_Algo_3(path = SAVE_PATH + str(i+Max)[1:], 
                     x_np = None, 
                     w_np = None,
                     y_np = None, 
                     y_np_range = None, 
                     x_hat_test = X_hat_test[i], 
                     w_hat_test = W_hat_test[i], 
                     x_range = None, 
                     x_hat_range = None, 
                     w_hat_range = None, clip=True)

# GAN Result
for i in range(len(X_Orig)):
    plt.subplot(4,len(X_Orig),k+1)
    plt.imshow(X_hat_test[i])
    plt.axis("Off")
    k=k+1

# Real
for i in range(len(X_Orig)):
    plt.subplot(4,len(X_Orig),k+1)
    plt.imshow(X_Orig[i])
    plt.axis("Off")
    k=k+1

plt.savefig('./results/'+"Comparison1.jpg")
plt.show()
