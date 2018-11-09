import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from PIL import Image
import numpy as np
import glob
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

batch_size = 128
epochs = 30

# input image dimensions
img_rows, img_cols = 38, 32

img_location = "percentages/"

data = []
labels = []

# the data, split between train and test sets
for f in glob.glob(img_location + '*.jpg'):
	for i in range(0,3):
		im = np.asarray(Image.open(f).convert('L').crop((132-66*i, 0,198-66*i,76)).resize((img_cols,img_rows)))
		if(not f[f.index('-')-i-1].isdigit()):
			num = 10
		else:
			num = f[f.index('-')-i-1]
		data.append(im)
		labels.append(num)


labels = np.array(labels).astype('float32')
# convert class vectors to binary class matrices
labels = keras.utils.to_categorical(labels, num_classes=11)


data = np.array(data).astype('float32')
data /= 255
data = data.reshape(data.shape[0], img_rows, img_cols, 1)

datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split=0.1)

datagen.fit(data)


model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows,img_cols,1)))
model.add(Conv2D(64,kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(11, activation='softmax'))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


filepath="models/weights-improvement-{epoch:02d}-{acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.fit_generator(
	datagen.flow(data, labels, batch_size = batch_size, subset='training'),
	steps_per_epoch=len(data) / batch_size * 2,
	epochs=epochs,
	verbose=1,
	callbacks=callbacks_list)
