import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

img=np.load("/home/sattam/FinalYearProject/Data/Train/Image/3.npy")
I=np.empty((1,128,128,1))
img=cv2.resize(img,(128,128),interpolation=cv2.INTER_CUBIC)
img=np.expand_dims(img,axis=2)
I[0,]=img


model=tf.keras.models.load_model("result.h5")
mask=model.predict(I)
print(mask)
try:
	mask=cv2.resize(mask[0],(128,128),interpolation=cv2.INTER_CUBIC)
except:
	print("hw")
plt.imshow(mask)
plt.show()

img=np.load("/home/sattam/FinalYearProject/Data/Train/Mask/3.npy")
plt.imshow(img)
plt.show()
