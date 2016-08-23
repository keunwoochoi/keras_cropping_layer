# keras_cropping_layer
This is a cropping layer - a customised layer implementation in keras.

# THIS LAYER IS MERGED IN KERAS SINCE 22 AUG 2016
* [It is merged.](https://github.com/fchollet/keras/pull/3509) Have fun!
* You can still check out the example code.

## What does it do?
It crops out input 2d images. 
`cropping.Cropping2D(cropping=((1,2),(3,4))` crops 1 pixel from top, 2 pixels from bottom, 3 pixels from left, and 4 pixels from right. 
In other words, it returns..
~~~python
return x[:, :, 1:-2, 3:4]
~~~
when `x` is the input as `(data_sample_idx, channel, height, width)`. 

## How to use
~~~python
import cropping	

model = Sequential()
model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='same',
                        input_shape=(1, img_rows, img_cols)))
model.add(cropping.Cropping2D(cropping=((2,2),(2,2)) ))
~~~

## What happens if run the example

~~~bash
$ python cnn_cropping.py
Using Theano backend.
Couldn't import dot_parser, loading of dot files will not be possible.
X_train shape: (60000, 1, 28, 28)
60000 train samples
10000 test samples
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
convolution2d_1 (Convolution2D)  (None, 1, 28, 28)     10          convolution2d_input_1[0][0]
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 1, 24, 24)     0           convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 1, 24, 24)     10          cropping2d_1[0][0]
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 1, 24, 24)     0           convolution2d_2[0][0]
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 1, 1, 1)       0           activation_1[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1)             0           maxpooling2d_1[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 10)            20          flatten_1[0][0]
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 10)            0           dense_1[0][0]
====================================================================================================
Total params: 40
____________________________________________________________________________________________________
Train on 60000 samples, validate on 10000 samples
Epoch 1/4
60000/60000 [==============================] - 6s - loss: 2.3120 - acc: 0.1088 - val_loss: 2.3031 - val_acc: 0.1144
Epoch 2/4
60000/60000 [==============================] - 4s - loss: 2.2946 - acc: 0.1278 - val_loss: 2.2758 - val_acc: 0.1918
Epoch 3/4
60000/60000 [==============================] - 3s - loss: 2.2423 - acc: 0.1833 - val_loss: 2.1945 - val_acc: 0.1857
Epoch 4/4
60000/60000 [==============================] - 3s - loss: 2.1588 - acc: 0.1910 - val_loss: 2.1191 - val_acc: 0.1900
Test score: 2.11912097359
Test accuracy: 0.19
~~~

## credit
[@ironbar](https://github.com/ironbar) [helped me debugging](https://github.com/fchollet/keras/issues/3162). 
