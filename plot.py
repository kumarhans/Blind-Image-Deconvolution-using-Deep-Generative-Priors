from PIL import Image
from glob import glob
import numpy as np
import os
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
Orig_Path_init = './results/SVHN/deblurring_video_alg1_init/*jpg'
Orig_Path = './results/SVHN/deblurring_video_alg1/*jpg'
Blurry_path = './data/blurVideoSVHN/*.jpg'

paths=[path for path in glob(Blurry_path)]
paths.sort()
Blurry_Images_orig = np.array([ imread(path) for path in paths]) / 255

# k=0
# count = 20
# nrow = 4
# ncol = 5
# for i in Blurry_Images_orig[-count:]:

#     plt.subplot(nrow,ncol,k+1)
#     plt.imshow(i)
#     plt.axis("Off")
#     k=k+1

# plt.savefig('./results/'+"Video.jpg")


# paths=[path for path in glob(Orig_Path_init)]
# paths.sort()

# paths = [path for path in paths if "Kernel" not in path and "_Algo1" in path]
# print(paths)

# Init = np.array([ imread(path) for path in paths]) / 255

# k=0
# for i in Init[-count:]:

#     plt.subplot(nrow,ncol,k+1)
#     plt.imshow(i)
#     plt.axis("Off")
#     k=k+1

# plt.savefig('./results/'+"Deblur_init.jpg")


# paths=[path for path in glob(Orig_Path)]
# paths.sort()

# paths = [path for path in paths if "Kernel" not in path and "_Algo1" in path]
# print(paths)

# Init = np.array([ imread(path) for path in paths]) / 255

# k=0
# for i in Init[-count:]:

#     plt.subplot(nrow,ncol,k+1)
#     plt.imshow(i)
#     plt.axis("Off")
#     k=k+1

# plt.savefig('./results/'+"Deblur.jpg")

Loss = np.load('./results/Loss_vs_iter.npy')

Loss = Loss[:,1000:]/80

mLoss = np.mean(Loss, axis = 0) 
plt.plot(mLoss,label='Algorithm 1')
plt.ylabel("Loss")
plt.xlabel("Iteration")
plt.savefig('./results/'+"1000Loss_vs_iter.jpg")


Loss = np.load('./results/Loss_vs_iter_init.npy')

Loss = Loss[:,1000:]/80

mLoss = np.mean(Loss, axis = 0) 
plt.plot(mLoss, label='Algorithm 2')
plt.ylabel("Loss")
plt.xlabel("Iteration")

plt.legend(loc="upper right")
plt.savefig('./results/'+"1000Loss_vs_iter_init.jpg")