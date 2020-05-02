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
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# paths
Orig_Path       = 'data/blurVideoSVHN/*.jpg'
Blur_Path       = './results/SVHN/Original Blurs/Test Blurs.npy'

# paths
REGULARIZORS = [1.0, 0.5, 100.0, 0.001]
LEARNING_RATE = 0.005
RANDOM_RESTARTS = 10
NOISE_STD       = 0.01
STEPS           = 6000
IMAGE_RANGE = [0,1]
optimizer       = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
def step_size(t):
    return 0.01 * np.exp( - t / 1000 )

SAVE_PATH       = './results/SVHN/deblurring_video_alg2/deblurring_'
PLOT_LOSS       = True
SAVE_RESULTS    = True
# -----------------------------------------------------------------------

# loading test blur kernels
W = np.load(Blur_Path) 
BLUR_RES = W.shape[1]

# loading svhn test images
paths=[path for path in glob(Orig_Path)]
paths.sort()
Blurry_Images_orig = np.array([ imread(path) for path in paths]) / 255
Blurry_Images_orig = Blurry_Images_orig[0:2]
IMAGE_RES = Blurry_Images_orig.shape[1]
CHANNELS = Blurry_Images_orig.shape[-1]

# loading svhn generator
SVHNGen = SVHNGenerator()
SVHNGen.GenerateModel()
SVHNGen.LoadWeights()
svhn_vae, svhn_encoder, svhn_decoder = SVHNGen.GetModels()
svhn_decoder.trainable = False
svhn_latent_dim = SVHNGen.latent_dim

# loading motion blur generator
BLURGen = MotionBlur()
BLURGen.GenerateModel()
BLURGen.LoadWeights()
blur_vae, blur_encoder, blur_decoder = BLURGen.GetModels()
blur_decoder.trainable = False
blur_latent_dim = BLURGen.latent_dim

# check if save dir exists, if not create a new one
try:
    os.stat(SAVE_PATH[:-12])
except:
    os.mkdir(SAVE_PATH[:-12])

#Converting to fourier form
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

# solving deconvolution using Algorithm 2
rr = np.shape(Blurry_Images)[0]
zi_tf = tf.Variable(tf.random_normal(shape=([rr, svhn_latent_dim])), dtype = 'float32')
zk_tf = tf.Variable(tf.random_normal(shape=([rr, blur_latent_dim])), dtype = 'float32')
x_tf  = tf.Variable(tf.random_normal(mean = 0.5, stddev  = 0.01,shape=([rr, IMAGE_RES,IMAGE_RES,CHANNELS])))

x_G  = svhn_decoder(zi_tf)
x_G = tf.reshape(x_G, shape=(rr,IMAGE_RES,IMAGE_RES,CHANNELS))
x_G = (x_G + 1)/2
y_fourier = tf.placeholder(shape=(rr, IMAGE_RES,IMAGE_RES,CHANNELS), dtype='complex64')

blur  = blur_decoder(zk_tf)
blur  = tf.reshape(blur, shape=(rr,BLUR_RES,BLUR_RES))
padding = np.int((IMAGE_RES -BLUR_RES)/2)
blur = tf.pad(blur, [[0,0], [padding,padding],[padding,padding]], 'CONSTANT')
blur_fourier = tf.fft2d( tf.cast(blur, dtype = 'complex64'))

# splitting tensors into 3 channels
y_fourier0 = y_fourier[:,:,:,0]; x_0 = x_tf[:,:,:,0]; x_G0 = x_G[:,:,:,0]
y_fourier1 = y_fourier[:,:,:,1]; x_1 = x_tf[:,:,:,1]; x_G1 = x_G[:,:,:,1]
y_fourier2 = y_fourier[:,:,:,2]; x_2 = x_tf[:,:,:,2]; x_G2 = x_G[:,:,:,2]

# 1st Channel Loss
x_0_fourier = tf.fft2d( tf.cast( x_0, dtype='complex64'))
loss_x0 = tf.reduce_mean( tf.square( tf.abs(y_fourier0 - x_0_fourier*blur_fourier) ), axis=[1,2])

x_Gi0_fourier = tf.fft2d( tf.cast( x_G0, dtype='complex64'))
loss_xG0 = tf.reduce_mean( tf.square( tf.abs(y_fourier0 - x_Gi0_fourier*blur_fourier) ), axis=[1,2])

# 2nd Channel Loss
x_1_fourier = tf.fft2d( tf.cast( x_1, dtype='complex64'))
loss_x1 = tf.reduce_mean( tf.square( tf.abs(y_fourier1 - x_1_fourier*blur_fourier) ), axis=[1,2])

x_Gi1_fourier = tf.fft2d( tf.cast( x_G1, dtype='complex64'))
loss_xG1 = tf.reduce_mean( tf.square( tf.abs(y_fourier1 - x_Gi1_fourier*blur_fourier) ), axis=[1,2])

# 3rd Channel Loss
x_2_fourier = tf.fft2d( tf.cast( x_2, dtype='complex64'))
loss_x2 = tf.reduce_mean( tf.square( tf.abs(y_fourier2 - x_2_fourier*blur_fourier) ), axis=[1,2])

x_Gi2_fourier = tf.fft2d( tf.cast( x_G2, dtype='complex64'))
loss_xG2 = tf.reduce_mean( tf.square( tf.abs(y_fourier2 - x_Gi2_fourier*blur_fourier) ), axis=[1,2])

Loss_xG_tf    = tf.constant(REGULARIZORS[0])*(loss_xG0 + loss_xG1 + loss_xG2)
Loss_x_tf     = tf.constant(REGULARIZORS[1])*(loss_x0 + loss_x1 + loss_x2)
x_minus_xG_tf = tf.constant(REGULARIZORS[2])*tf.reduce_mean( tf.square( tf.abs(x_tf - x_G)), axis=[1,2,3])
LossTV_tf     = tf.constant(REGULARIZORS[3])*tf.image.total_variation(x_tf)
TotalLoss_tf  = Loss_xG_tf + Loss_x_tf + x_minus_xG_tf + LossTV_tf

opt =  optimizer.minimize(TotalLoss_tf, var_list = [zi_tf, zk_tf, x_tf])
sess = K.get_session()
sess.run(tf.variables_initializer([zi_tf, zk_tf, x_tf]))
Losses = []

# running optimizer steps
for i in tqdm(range(STEPS), ascii=True, desc = 'Solving Deconv.'):
    losses = sess.run([opt, TotalLoss_tf, Loss_xG_tf, Loss_x_tf, x_minus_xG_tf], 
                      feed_dict = {y_fourier: Blurry_Images})
    Losses.append([loss for loss in losses[1:] ])
Losses = np.array(Losses)
zi_hat, zk_hat, x_hat = sess.run([zi_tf, zk_tf, x_tf])

tmp = []
for i in range(4):
    tmp.append( [loss[i] for loss in Losses])
Losses = tmp
TotalLoss, Loss_xG, Loss_x, x_minus_xG = Losses

# convergence plots 
if PLOT_LOSS:
    plt.figure(figsize=(10,5))
    plt.subplot(2,2,1)
    plt.plot(np.mean(TotalLoss, axis=1)); plt.title('Total Loss')
    plt.subplot(2,2,2)
    plt.plot(np.mean(Loss_x, axis=1)); plt.title('x Loss')
    plt.subplot(2,2,3)
    plt.plot(np.mean(Loss_xG, axis=1)); plt.title('xG Loss')
    plt.subplot(2,2,4)
    plt.plot(np.mean(x_minus_xG, axis=1)); plt.title('x - xG')
    plt.show()

# extracting best images from random restarts with minimum residual error
X_Hat = []
XG_Hat   = []
W_Hat = []
for i in range(len(Blurry_Images_orig)):
    x_i      =      Blurry_Images_orig[i]
    zi_hat_i = zi_hat[i*RANDOM_RESTARTS:(i+1)*RANDOM_RESTARTS]
    zk_hat_i = zk_hat[i*RANDOM_RESTARTS:(i+1)*RANDOM_RESTARTS]
    x_hat_i    = x_hat[i*RANDOM_RESTARTS:(i+1)*RANDOM_RESTARTS]
    w_hat_i    = blur_decoder.predict(zk_hat_i)[:,:,:,0]
    x_hat_i      = np.clip(x_hat_i, 0, 1)
    loss_i       = [ComputeResidual(Y_np[i], x, w) for x,w in zip(x_hat_i,w_hat_i)]
    min_loss_loc = np.argmin(loss_i)
    
    zi_hat_recov = zi_hat_i[min_loss_loc].reshape([1,svhn_latent_dim])
    zk_hat_recov = zk_hat_i[min_loss_loc].reshape([1,blur_latent_dim])
    x_hat_recov  = x_hat_i[min_loss_loc] 
    w_hat = blur_decoder.predict(zk_hat_recov).reshape(BLUR_RES,BLUR_RES)
    xg_hat = svhn_decoder.predict(zi_hat_recov).reshape(IMAGE_RES,IMAGE_RES,CHANNELS)
    X_Hat.append(x_hat_recov); W_Hat.append(w_hat); XG_Hat.append(xg_hat)
X_Hat = np.array(X_Hat)
W_Hat = np.array(W_Hat)
XG_Hat = np.array(XG_Hat)

# normalizing images
X_Hat = np.clip(X_Hat, 0,1)
XG_Hat = (XG_Hat + 1)/2

# calculating psnr and ssim --- in paper both PSNR and SSIM where computed
# using matlab
# PSNR = []; SSIM = []
# for x, x_pred in zip(X_Orig, X_Hat):
#     psnr = compare_psnr(x, x_pred.astype('float64'))
#     ssim = compare_ssim(x, x_pred.astype('float64'), multichannel=True)
#     PSNR.append(psnr); SSIM.append(ssim)
# print("PSNR = ", np.mean(PSNR))
# print("SSIM = ", np.mean(SSIM))


# saving results
Max = 10**len(str(len(Blurry_Images_orig)-1))
if SAVE_RESULTS:
    for i in range(len(Blurry_Images_orig)):
        Save_Results(path = SAVE_PATH + str(i+Max)[1:], 
                         x_np = Blurry_Images_orig[i], 
                         w_np = None,
                         y_np = Y_np[i], 
                         y_np_range = None , 
                         x_hat_test = X_Hat[i], 
                         w_hat_test = W_Hat[i], 
                         x_range = None, 
                         x_hat_range = XG_Hat[i], 
                         w_hat_range = None, clip=True)