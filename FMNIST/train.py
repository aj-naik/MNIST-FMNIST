import keras
from keras import models
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Dropout
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.saving.save import load_model 

(x_train,y_train), (x_test,y_test) = fashion_mnist.load_data()

# print(x_train.shape)

x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)
input_shape = (28,28,1)

y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

x_train = (x_train.astype('float32'))/255
x_test = (x_test.astype('float32'))/255

# print(x_train.shape)

batch_size = 128
num_classes = 10
epochs = 20

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.2))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(rate=0.4))
model.add(Dense(num_classes,activation='softmax'))

model.compile(keras.optimizers.Adam(),keras.losses.categorical_crossentropy,metrics=['accuracy'])

fitting = model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test))

print('Training Completed')

model.save('fmnist.h5')

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])








