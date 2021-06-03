# MNIST-FMNIST
Tkinter based apps for classification of numbers and clothes using the MNIST and FMNIST Datasets. 

# Model Info
- Used keras for creating models.
- Used 2 Conv2D layers with kernal size (3,3), 'Relu' activation and 32, 64 filters respectively.
- Added a MaxPool2D layer with pool size (2,2), A couple of Droput layers, Dense layers with activations 'Relu' and 'softmax for final dense layers with 10 units for 10 different classes.
- Used Adam optimizer and categorical_crossentropy loss and selected the metric as accuracy.

# Working 
- After training the model, it is loaded into the app. 
- The app has a canvas where the user can draw. 
- The drawing from canvas is then captured and the image is processed before being passed through the model.
- The model then classifies the image passed and the app prints the class predicted along with the accuracy of prediction on screen.

# Demo
![Demo](https://user-images.githubusercontent.com/51918054/120610547-ead2d900-c470-11eb-84e1-f356a4fbd3a2.gif)

As can be seen from the demo, the model does not always classify accurately. Tinkering with model layers might help improve accuracy

# Running it
- Install all python dependencies like Keras, Pillow, Tensorflow, Win32gui, Numpy
- If you want to train the model, Edit train.py to your requirments and run it.
- To start the app, run app.py from CMD or through IDLE. 

# Project History
Created this because I wanted a gui for digit and cloth classification instead of using a jupyter notebook. Used Tkinter because it is easy enough to use. Project works but accuracy needs to be  improved. Quite a few classifications are wrong. This is one of my first deep learning gui project.
