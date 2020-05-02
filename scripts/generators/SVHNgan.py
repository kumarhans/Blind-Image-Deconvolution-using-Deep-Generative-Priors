
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, BatchNormalization, Activation,LeakyReLU, Dense, Conv2DTranspose, Input, Lambda, Reshape, Flatten, UpSampling2D, MaxPooling2D
from keras.models import Model, Sequential
import keras.backend as K
from keras import initializers
import tensorflow as tf

class SVHNganGenerator():

    def __init__(self):
        self.latent_dim = 100        # Dimension of Latent Representation
        self.GAN = None
        self.weights_path = './model weights/gan_hans_svhn_extra.h5'

         
    def GenerateModel(self):
        gf_dim = 64
        gan = Sequential()
        gan.add(Dense(4*4*512, input_shape=(100,)))
        gan.add(Reshape([4,4,512]))
        gan.add(BatchNormalization())
        gan.add(LeakyReLU(alpha=.2))
        gan.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding='same'))  
        gan.add(BatchNormalization())
        gan.add(LeakyReLU(alpha=.2))
        gan.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same')) 
        gan.add(BatchNormalization())
        gan.add(LeakyReLU(alpha=.2))
        gan.add(Conv2DTranspose(3, kernel_size=5, strides=2, padding='same'))
        gan.add(Activation('tanh')) 

        self.GAN = gan


 


    def LoadWeights(self):
        self.GAN.load_weights(self.weights_path)

    def GetModels(self):
        return self.GAN

if __name__ == '__main__':
    Gen = SVHNganGenerator()
    Gen.GenerateModel()
    Gen.weights_path = '../model weights/gan_hans_svhn_extra.h5'
    Gen.LoadWeights()
    gan = Gen.GetModels()
    
    n_samples = 10
    len_z = Gen.latent_dim
    z = np.random.normal(0,1,size=(n_samples*n_samples ,len_z))
    sampled = gan.predict(z)
    sampled = (sampled+1)/2
    
    k = 0
    for i in range(n_samples):
        for j in range(n_samples):
            img = sampled[k]
            plt.subplot(n_samples,n_samples,k+1)
            plt.imshow(img)
            plt.axis("Off")
            k=k+1
    plt.show()