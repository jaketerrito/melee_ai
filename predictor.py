import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from PIL import Image
import numpy as np

img_rows, img_cols = 76//2, 66//2

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows,img_cols,1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(11, activation='softmax'))
model.load_weights("NumberBlank.hdf5")
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

import mss
import time

import pyautogui as pg
while True:
	num = ''
	for i in range(0,3):
		mon = {"top": 886, "left": 521-i*66, "width": 66, "height": 76}
		sct = mss.mss()
		im = sct.grab(mon)
		im= Image.frombytes("RGB", im.size, im.bgra, "raw", "BGRX")
		im = np.asarray(im.convert('L').resize((img_cols,img_rows)))
		im = np.array(im).astype('float32')
		im /= 255
		im = im.reshape(1, img_rows, img_cols, 1)
		guess = model.predict(im)
		guess = list(guess[0])
		if(guess.index(max(guess)) != 10):
			num =str(guess.index(max(guess))) + num
	print(num)
