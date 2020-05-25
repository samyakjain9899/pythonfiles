from keras.datasets import mnist
dataset = mnist.load_data('mymnist.db')
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
train , test = dataset
len(train)

X_train ,Y_train = train
X_train.shape

X_test , Y_test = test
X_test.shape

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train/=255
X_test/=255

from keras.utils.np_utils import to_categorical



number_of_classes = 10

Y_train = to_categorical(Y_train, number_of_classes)
Y_test = to_categorical(Y_test, number_of_classes)


model = Sequential()

model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                   input_shape=(28, 28, 1)
                       ))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                       ))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())



model.add(Dense(units=128, activation='relu'))

model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras_preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)

test_datagen = ImageDataGenerator()

train_set= train_datagen.flow(X_train, Y_train, batch_size=64)
test_set = test_datagen.flow(X_test, Y_test, batch_size=64)

model.fit(train_set, steps_per_epoch=60000//64, epochs=10, 
                    validation_data=test_set, validation_steps=10000//64)
scores = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])