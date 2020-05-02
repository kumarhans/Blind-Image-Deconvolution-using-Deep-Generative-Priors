import tensorflow as tf
import keras.backend as K
from Utils import *
from generators.MotionBlurGenerator import *
from generators.SVHNGenerator import *
K.set_learning_phase(0)
from glob import glob
import os
import numpy as np
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import time
start_time = time.time()



# paths
# Orig_Path       = './results/hansSVHN/BlurSeq/*.png'
Orig_Path       = './data/blurVideoSVHN/*.jpg'
Clean_Path      =   './results/hansSVHN/CleanSeq/*.png'
Range_Path      = './results/SVHN/Range Images/*.png'
Blur_Path       = './results/SVHN/Original Blurs/Test Blurs.npy'

# paths
REGULARIZORS = [0.01 , 0.01]
RANDOM_RESTARTS = 10
NOISE_STD       = 0.01
STEPS           = 6000
IMAGE_RANGE = [0,1]

def step_size(t):
    return 0.01 * np.exp( - t / 1000 )

SAVE_PATH       = './results/SVHN/deblurring_video_alg1/deblurring_'
# -----------------------------------------------------------------------

# loading test blur kernels
W = np.load(Blur_Path) 
BLUR_RES = W.shape[1]

paths=[path for path in glob(Orig_Path)]
paths.sort()
Blurry_Images_orig = np.array([ imread(path) for path in paths]) / 255

paths=[path for path in glob(Clean_Path)]
paths.sort()
X_Orig = np.array([ imread(path) for path in paths]) / 255

print(Blurry_Images_orig.shape)
Blurry_Images_orig = Blurry_Images_orig[-20:]
X_Orig = X_Orig[0:2]

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

# check if save dir exists, if not create a new one
try:
    os.stat(SAVE_PATH[:-12])
except:
    os.mkdir(SAVE_PATH[:-12])

# generating blurry images from test #COnverting to fourier form
Y_np = []
Blurry_Images = []
for i in tqdm(range(len(Blurry_Images_orig)), ascii=True, desc ='Gen-Test-Blurry'):
    b_np = Blurry_Images_orig[i]
    y_np, y_f = toFourier(b_np, noise_std = NOISE_STD )
    Y_np.append(y_np)
    for _ in range(RANDOM_RESTARTS):
        Blurry_Images.append(y_f)

Y_np = np.array(Y_np)
Blurry_Images = np.array(Blurry_Images)

# alternating gradient descent for test images
Loss = np.zeros((0,6000))
h_hat = np.zeros((0,50))
m_hat = np.zeros((0,100))

for i in range(len(Blurry_Images_orig)):
    print(i)
    image_gradients, blur_gradients, get_loss = Generate_Gradient_Functions(rr = 10,
                                                                        reg = REGULARIZORS, image_range = IMAGE_RANGE,
                                                                        decoder = svhn_decoder, blur_decoder = blur_decoder,
                                                                        image_res = IMAGE_RES, blur_res = BLUR_RES,
                                                                        channels = CHANNELS)
    
    m_hat_i, h_hat_i, Loss_i = Optimize_Parallel(blurry_fourier = Blurry_Images[i*RANDOM_RESTARTS:(i+1)*RANDOM_RESTARTS], stepsize=step_size,steps = STEPS,
                                        image_grad = image_gradients , blur_grad = blur_gradients, 
                                        getloss = get_loss, latent_image_dim = svhn_latent_dim , latent_blur_dim = blur_latent_dim)
    
    m_hat = np.vstack((m_hat,m_hat_i))
    h_hat = np.vstack((h_hat,h_hat_i))
    Loss = np.vstack((Loss,Loss_i))

    print('m',m_hat.shape)
    print('h',h_hat.shape)

    # # initial_m = m_hat_i
    # initial_h = h_hat_i
    
print(Loss.shape)

X_hat_test = []
W_hat_test = []
Total_loss = 0
for i in range(len(Blurry_Images_orig)):
    m_hat_i = m_hat[i*RANDOM_RESTARTS:(i+1)*RANDOM_RESTARTS]
    h_hat_i = h_hat[i*RANDOM_RESTARTS:(i+1)*RANDOM_RESTARTS]
    Loss_i  =  Loss[i*RANDOM_RESTARTS:(i+1)*RANDOM_RESTARTS]
    x_hat_test, w_hat_test, loss_last_iter_test = Get_Min_Loss(Loss_i, m_hat_i, h_hat_i, decoder = svhn_decoder, blur_decoder = blur_decoder,
                                                               latent_image_dim = svhn_latent_dim, latent_blur_dim = blur_latent_dim,  print_grad=False)
    X_hat_test.append(x_hat_test)
    W_hat_test.append(w_hat_test)
    Total_loss += loss_last_iter_test


X_hat_test = np.array(X_hat_test)
W_hat_test = np.array(W_hat_test)

print("Done")
print("X_hat_test", X_hat_test.shape)
print("W_hat_test", W_hat_test.shape)

print("--- %s seconds ---" % (time.time() - start_time))
print("Average Loss %s" % (Total_loss/len(Blurry_Images_orig)))

# saving results
Max = 10**6
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

# calculating psnr and ssim --- in paper both PSNR and SSIM where computed
# using matlab
X_hat_test = np.array(X_hat_test)
X_hat_test = np.clip(X_hat_test, 0,1)

# PSNR = []; SSIM = []
# for x, x_pred in zip(X_Orig, X_hat_test):
#     psnr = compare_psnr(x, x_pred.astype('float64'))
#     ssim = compare_ssim(x, x_pred.astype('float64'), multichannel=True)
#     PSNR.append(psnr); SSIM.append(ssim)
# print("PSNR = ", np.mean(PSNR))
# print("SSIM = ", np.mean(SSIM))

np.save('./results/Loss_vs_iter.npy', Loss)

Loss = np.mean(Loss, axis = 0) 
plt.plot(Loss)
plt.ylabel("Loss")
plt.xlabel("Iteration")
plt.savefig('./results/'+"Loss_vs_iter.jpg")
