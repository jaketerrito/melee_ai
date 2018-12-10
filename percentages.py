import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from PIL import Image
import numpy as np
import mss
import time

def get_percents(sct, s_w, s_h, player):
    img_rows, img_cols = 38, 32
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(img_rows,img_cols,1)))
    model.add(Conv2D(64,kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(11, activation='softmax'))
    model.load_weights("HealthPredWeights.hdf5")
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    offset = [389, 527][player]
    while True:
        num = ''
        for i in range(0,3):
            mon = {"top": s_h-74, "left": s_w-offset-(i*32), "width": 32, "height": 38}
            im = sct.grab(mon)
            im= Image.frombytes("RGB", im.size, im.bgra, "raw", "BGRX")
            im = np.asarray(im.convert('L'))
            im = np.array(im).astype('float32')
            im /= 255
            im = im.reshape(1, img_rows, img_cols, 1)
            guess = model.predict(im)
            guess = list(guess[0])
            if(guess.index(max(guess)) != 10):
                num =str(guess.index(max(guess))) + num
        if num == '':
            yield(10000)
        else:
            yield(int(num))
