import tensorflow as tf
import keras.backend as K
import numpy as np
from Utils import *
from generators.MotionBlurGenerator import *
from generators.SVHNgan import SVHNganGenerator
K.set_learning_phase(0)
from glob import glob
import os
from PIL import Image
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"


# paths
Orig_Path       = './results/hansSVHN/Original Images/*.png'
Blur_Path       = './results/hansSVHN/Original Blurs/Test Blurs.npy'

# constants
REGULARIZORS = [0.01 , 0.01]
RANDOM_RESTARTS = 10
NOISE_STD       = 0.01
STEPS           = 10000
IMAGE_RANGE = [-1,1]
def step_size(t):
    return 0.01 * np.exp( - t / 1000 )

# loading test blur images
W = np.load(Blur_Path) 
BLUR_RES = W.shape[1]

# loading svhn test images
X_Orig = []
for filename in glob(Orig_Path):
    im = Image.open(filename)
    im = np.array(im)
    X_Orig.append(im)
X_Orig = np.array(X_Orig) / 255
X_Orig = X_Orig[0:10]
IMAGE_RES = X_Orig.shape[1]
CHANNELS = X_Orig.shape[-1]

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

# generating blurry images from range
Blurry_Images_range = []
Y_np_range = []
for i in tqdm(range(len(X_Orig)), ascii=True, desc ='Gen-Range-Blurry'):
    y_np, y_f = GenerateBlurry(X_Orig[i], W[i], noise_std = NOISE_STD )
    Y_np_range.append(y_np)
    for _ in range(RANDOM_RESTARTS):
        Blurry_Images_range.append(y_f)

Y_np_range = np.array(Y_np_range)
Blurry_Images_range = np.array(Blurry_Images_range)


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

k = 0

# Blurred
for i in range(len(X_Orig)):
    plt.subplot(4,len(X_Orig),k+1)
    plt.imshow(Y_np[i])
    plt.axis("Off")
    k=k+1

# GAN Algo 1 Result
for i in range(len(X_Orig)):
    plt.subplot(4,len(X_Orig),k+1)
    plt.imshow(X_hat_test[i])
    plt.axis("Off")
    k=k+1


# algorithm constants
REGULARIZORS  = [1.0, 0.5, 100.0, 0.001]
LEARNING_RATE = 0.005
optimizer     = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

# loading svhn generator
SVHNGen = SVHNganGenerator()
SVHNGen.GenerateModel()
SVHNGen.LoadWeights()
SVHNGAN = SVHNGen.GetModels()
SVHNGAN.trainable = False
svhn_latent_dim = SVHNGen.latent_dim

# loading motion blur generator
BLURGen = MotionBlur()
BLURGen.GenerateModel()
BLURGen.LoadWeights()
blur_vae, blur_encoder, blur_decoder = BLURGen.GetModels()
blur_decoder.trainable = False
blur_latent_dim = BLURGen.latent_dim

# solving deconvolution using Algorithm 2
rr = Blurry_Images.shape[0]
zi_tf = tf.Variable(tf.random_normal(shape=([rr, svhn_latent_dim])), dtype = 'float32')
zk_tf = tf.Variable(tf.random_normal(shape=([rr, blur_latent_dim])), dtype = 'float32')
x_tf  = tf.Variable(tf.random_normal(mean = 0.5, stddev  = 0.01,shape=([rr, IMAGE_RES,IMAGE_RES,CHANNELS])))

x_G = SVHNGAN(zi_tf)
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

# extracting best images from random restarts with minimum residual error
X_Hat = []
XG_Hat   = []
W_Hat = []
for i in range(len(X_Orig)):
    x_i      =      X_Orig[i]
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
    xg_hat = SVHNGAN.predict(zi_hat_recov).reshape(IMAGE_RES,IMAGE_RES,CHANNELS)
    X_Hat.append(x_hat_recov); W_Hat.append(w_hat); XG_Hat.append(xg_hat)
X_Hat = np.array(X_Hat)
W_Hat = np.array(W_Hat)
XG_Hat = np.array(XG_Hat)

# normalizing images
X_Hat = np.clip(X_Hat, 0,1)
XG_Hat = (XG_Hat + 1)/2

# GAN Algo 2 Result
for i in range(len(X_Orig)):
    plt.subplot(4,len(X_Orig),k+1)
    plt.imshow(XG_Hat[i])
    plt.axis("Off")
    k=k+1

# Real
for i in range(len(X_Orig)):
    plt.subplot(4,len(X_Orig),k+1)
    plt.imshow(X_Orig[i])
    plt.axis("Off")
    k=k+1

plt.savefig('./results/'+"Comparison_GAN.jpg")
plt.show()