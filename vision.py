from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import keras.utils.np_utils as utils
from keras.constraints import maxnorm
from tensorflow.keras.optimizers import SGD

model = Sequential()

lables = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)

# input_shape is 32 x 32 pixels x 3 different colours (rgb)
# padding = 'same' makes sure that the output image of this convolution doesn't shrink
# kernel_constraint proportionately decreses all of the values 
# so that the highest value is 3
model.add(Conv2D(filters=32, kernel_size=(3, 3), 
                 input_shape=(32, 32, 3), 
                 activation='relu', padding='same', 
                 kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) #flattens array
model.add(Dense(units=512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(rate=0.5)) #prevents over-fitting (only applied during training, not testing)
# used to produce output of 10 categories and their probabilities (softmax) -> final layer
model.add(Dense(units=10, activation='softmax')) 
model.compile(optimizer=SGD(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=x_train, y=y_train, epochs=10, batch_size=32)