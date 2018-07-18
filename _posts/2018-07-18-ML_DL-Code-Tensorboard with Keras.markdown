---
layout: post
title: Tensorboard with Keras
date: 2018-07-18 03:09:00
img: ML_DL/keras.png # Add image post (optional)
categories: [ML_DL Code] 
tags: [Tensorboard, Keras] # add tag
---

+ github : [Tensorboard with Kearas](https://nbviewer.jupyter.org/github/gaussian37/Deep-Learning/blob/master/Library/Keras/Keras%20Reference/Tensorboard%20with%20Keras.ipynb)

Tensorflow offers Tensorboard which is good monitoring tool of learning procedure. If you run Keras with Tensorflow as backend, you can use it. Accordingly, first, you need to set the Tensorflow as backend at the setting file "keras.json". 

    vi ~/.keras/keras.json

In the "keras.json", You must set the "backed" : "tensorflow".

### Let's run Keras sample !


```python
import keras
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
```

    Using TensorFlow backend.
    




    ['/job:localhost/replica:0/task:0/device:GPU:0']



set the random seed


```python
np.random.seed(3)
```

load train & test set


```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```


```python
print(x_train.shape, x_test.shape)
```

    (60000, 28, 28) (10000, 28, 28)
    

split the train & test set


```python
x_val = x_train[50000:]
y_val = y_train[50000:]
x_train = x_train[:50000]
y_train = y_train[:50000]
```

Data preprocessing


```python
x_train = x_train.reshape(50000, 784).astype("float32") / 255.0
x_val = x_val.reshape(10000, 784).astype("float32") / 255.0
x_test = x_test.reshape(10000, 784).astype("float32") / 255.0
```

Transform label data to one-hot encoding


```python
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)
y_test = np_utils.to_categorical(y_test)
```

Set the model


```python
model = Sequential()
model.add(Dense(units = 2, input_dim = 28*28, activation="relu"))
model.add(Dense(units = 10, activation="softmax"))
```

Set loss, optimize, metrics


```python
model.compile(loss = "categorical_crossentropy", optimizer = "sgd", metrics = ["accuracy"])
```

Learning the model.

### Use Tensorboard now !


```python
tb_hist = keras.callbacks.TensorBoard("./graph", histogram_freq= 0, write_graph = True, write_images=True)
```


```python
model.fit(x_train, y_train, epochs=100, batch_size=10, validation_data=(x_val, y_val), callbacks=[tb_hist])
```

    Train on 50000 samples, validate on 10000 samples
    Epoch 1/100
    50000/50000 [==============================] - 17s 341us/step - loss: 1.5501 - acc: 0.4055 - val_loss: 1.2844 - val_acc: 0.4816
    Epoch 2/100
    50000/50000 [==============================] - 16s 321us/step - loss: 1.2276 - acc: 0.5280 - val_loss: 1.1450 - val_acc: 0.5678
    Epoch 3/100
    50000/50000 [==============================] - 16s 317us/step - loss: 1.1538 - acc: 0.5796 - val_loss: 1.0961 - val_acc: 0.6091
    Epoch 4/100
    50000/50000 [==============================] - 16s 328us/step - loss: 1.1111 - acc: 0.6230 - val_loss: 1.0570 - val_acc: 0.6489
    Epoch 5/100
    50000/50000 [==============================] - 16s 327us/step - loss: 1.0740 - acc: 0.6504 - val_loss: 1.0153 - val_acc: 0.6742
    Epoch 6/100
    50000/50000 [==============================] - 16s 315us/step - loss: 1.0490 - acc: 0.6639 - val_loss: 0.9967 - val_acc: 0.6758
    Epoch 7/100
    50000/50000 [==============================] - 16s 312us/step - loss: 1.0299 - acc: 0.6717 - val_loss: 0.9925 - val_acc: 0.6777
    Epoch 8/100
    50000/50000 [==============================] - 16s 324us/step - loss: 1.0158 - acc: 0.6767 - val_loss: 0.9686 - val_acc: 0.6908
    Epoch 9/100
    50000/50000 [==============================] - 16s 323us/step - loss: 1.0059 - acc: 0.6803 - val_loss: 0.9679 - val_acc: 0.6781
    Epoch 10/100
    50000/50000 [==============================] - 16s 330us/step - loss: 0.9977 - acc: 0.6825 - val_loss: 0.9541 - val_acc: 0.6957
    Epoch 11/100
    50000/50000 [==============================] - 16s 326us/step - loss: 0.9926 - acc: 0.6848 - val_loss: 0.9549 - val_acc: 0.6927
    Epoch 12/100
    50000/50000 [==============================] - 16s 323us/step - loss: 0.9861 - acc: 0.6850 - val_loss: 0.9424 - val_acc: 0.6901
    Epoch 13/100
    50000/50000 [==============================] - 17s 334us/step - loss: 0.9838 - acc: 0.6870 - val_loss: 0.9394 - val_acc: 0.6997
    Epoch 14/100
    50000/50000 [==============================] - 17s 333us/step - loss: 0.9798 - acc: 0.6871 - val_loss: 0.9461 - val_acc: 0.6956
    Epoch 15/100
    50000/50000 [==============================] - 16s 323us/step - loss: 0.9766 - acc: 0.6880 - val_loss: 0.9313 - val_acc: 0.6974
    Epoch 16/100
    50000/50000 [==============================] - 16s 327us/step - loss: 0.9725 - acc: 0.6890 - val_loss: 0.9380 - val_acc: 0.6909
    Epoch 17/100
    50000/50000 [==============================] - 16s 318us/step - loss: 0.9719 - acc: 0.6890 - val_loss: 0.9362 - val_acc: 0.6932
    Epoch 18/100
    50000/50000 [==============================] - 16s 325us/step - loss: 0.9698 - acc: 0.6887 - val_loss: 0.9381 - val_acc: 0.6941
    Epoch 19/100
    50000/50000 [==============================] - 16s 317us/step - loss: 0.9689 - acc: 0.6906 - val_loss: 0.9181 - val_acc: 0.7064
    Epoch 20/100
    50000/50000 [==============================] - 16s 323us/step - loss: 0.9656 - acc: 0.6915 - val_loss: 0.9401 - val_acc: 0.6894
    Epoch 21/100
    50000/50000 [==============================] - 16s 315us/step - loss: 0.9651 - acc: 0.6904 - val_loss: 0.9358 - val_acc: 0.6999
    Epoch 22/100
    50000/50000 [==============================] - 16s 325us/step - loss: 0.9632 - acc: 0.6938 - val_loss: 0.9270 - val_acc: 0.7024
    Epoch 23/100
    50000/50000 [==============================] - 16s 319us/step - loss: 0.9627 - acc: 0.6912 - val_loss: 0.9302 - val_acc: 0.6969
    Epoch 24/100
    50000/50000 [==============================] - 16s 317us/step - loss: 0.9605 - acc: 0.6914 - val_loss: 0.9230 - val_acc: 0.7000
    Epoch 25/100
    50000/50000 [==============================] - 16s 325us/step - loss: 0.9599 - acc: 0.6932 - val_loss: 0.9214 - val_acc: 0.6946
    Epoch 26/100
    50000/50000 [==============================] - 15s 308us/step - loss: 0.9576 - acc: 0.6919 - val_loss: 0.9300 - val_acc: 0.6955
    Epoch 27/100
    50000/50000 [==============================] - 16s 321us/step - loss: 0.9580 - acc: 0.6926 - val_loss: 0.9241 - val_acc: 0.7010
    Epoch 28/100
    50000/50000 [==============================] - 16s 322us/step - loss: 0.9562 - acc: 0.6910 - val_loss: 0.9154 - val_acc: 0.7003
    Epoch 29/100
    50000/50000 [==============================] - 16s 326us/step - loss: 0.9561 - acc: 0.6938 - val_loss: 0.9158 - val_acc: 0.7016
    Epoch 30/100
    50000/50000 [==============================] - 17s 330us/step - loss: 0.9548 - acc: 0.6927 - val_loss: 0.9217 - val_acc: 0.6988
    Epoch 31/100
    50000/50000 [==============================] - 16s 322us/step - loss: 0.9545 - acc: 0.6952 - val_loss: 0.9175 - val_acc: 0.7066
    Epoch 32/100
    50000/50000 [==============================] - 16s 329us/step - loss: 0.9527 - acc: 0.6942 - val_loss: 0.9126 - val_acc: 0.7088
    Epoch 33/100
    50000/50000 [==============================] - 16s 320us/step - loss: 0.9530 - acc: 0.6954 - val_loss: 0.9164 - val_acc: 0.6984
    Epoch 34/100
    50000/50000 [==============================] - 16s 325us/step - loss: 0.9513 - acc: 0.6951 - val_loss: 0.9238 - val_acc: 0.6923
    Epoch 35/100
    50000/50000 [==============================] - 16s 325us/step - loss: 0.9509 - acc: 0.6959 - val_loss: 0.9103 - val_acc: 0.7003
    Epoch 36/100
    50000/50000 [==============================] - 16s 330us/step - loss: 0.9499 - acc: 0.6951 - val_loss: 0.9296 - val_acc: 0.6919
    Epoch 37/100
    50000/50000 [==============================] - 16s 320us/step - loss: 0.9501 - acc: 0.6960 - val_loss: 0.9109 - val_acc: 0.7083
    Epoch 38/100
    50000/50000 [==============================] - 16s 325us/step - loss: 0.9492 - acc: 0.6976 - val_loss: 0.9170 - val_acc: 0.7025
    Epoch 39/100
    50000/50000 [==============================] - 16s 325us/step - loss: 0.9487 - acc: 0.6985 - val_loss: 0.9176 - val_acc: 0.7017
    Epoch 40/100
    50000/50000 [==============================] - 16s 327us/step - loss: 0.9478 - acc: 0.6963 - val_loss: 0.9121 - val_acc: 0.7011
    Epoch 41/100
    50000/50000 [==============================] - 16s 323us/step - loss: 0.9467 - acc: 0.6969 - val_loss: 0.9116 - val_acc: 0.7038
    Epoch 42/100
    50000/50000 [==============================] - 16s 325us/step - loss: 0.9460 - acc: 0.6986 - val_loss: 0.9014 - val_acc: 0.7127
    Epoch 43/100
    50000/50000 [==============================] - 16s 320us/step - loss: 0.9460 - acc: 0.6984 - val_loss: 0.9029 - val_acc: 0.7110
    Epoch 44/100
    50000/50000 [==============================] - 17s 330us/step - loss: 0.9441 - acc: 0.6997 - val_loss: 0.9065 - val_acc: 0.7086
    Epoch 45/100
    50000/50000 [==============================] - 16s 312us/step - loss: 0.9450 - acc: 0.6972 - val_loss: 0.9233 - val_acc: 0.7008
    Epoch 46/100
    50000/50000 [==============================] - 15s 308us/step - loss: 0.9440 - acc: 0.6981 - val_loss: 0.9112 - val_acc: 0.7054
    Epoch 47/100
    50000/50000 [==============================] - 16s 324us/step - loss: 0.9439 - acc: 0.6986 - val_loss: 0.9004 - val_acc: 0.7080
    Epoch 48/100
    50000/50000 [==============================] - 16s 329us/step - loss: 0.9430 - acc: 0.6998 - val_loss: 0.9004 - val_acc: 0.7122
    Epoch 49/100
    50000/50000 [==============================] - 17s 331us/step - loss: 0.9422 - acc: 0.7002 - val_loss: 0.9037 - val_acc: 0.7039
    Epoch 50/100
    50000/50000 [==============================] - 16s 321us/step - loss: 0.9417 - acc: 0.6984 - val_loss: 0.9131 - val_acc: 0.7127
    Epoch 51/100
    50000/50000 [==============================] - 17s 333us/step - loss: 0.9416 - acc: 0.7005 - val_loss: 0.9037 - val_acc: 0.7049
    Epoch 52/100
    50000/50000 [==============================] - 16s 322us/step - loss: 0.9415 - acc: 0.7013 - val_loss: 0.9067 - val_acc: 0.7094
    Epoch 53/100
    50000/50000 [==============================] - 16s 312us/step - loss: 0.9403 - acc: 0.6998 - val_loss: 0.9042 - val_acc: 0.7118
    Epoch 54/100
    50000/50000 [==============================] - 16s 321us/step - loss: 0.9394 - acc: 0.7013 - val_loss: 0.8980 - val_acc: 0.7114
    Epoch 55/100
    50000/50000 [==============================] - 16s 323us/step - loss: 0.9399 - acc: 0.7023 - val_loss: 0.8991 - val_acc: 0.7123
    Epoch 56/100
    50000/50000 [==============================] - 16s 322us/step - loss: 0.9392 - acc: 0.7023 - val_loss: 0.9102 - val_acc: 0.7098
    Epoch 57/100
    50000/50000 [==============================] - 16s 323us/step - loss: 0.9391 - acc: 0.7032 - val_loss: 0.8952 - val_acc: 0.7102
    Epoch 58/100
    50000/50000 [==============================] - 16s 328us/step - loss: 0.9389 - acc: 0.7014 - val_loss: 0.9045 - val_acc: 0.7053
    Epoch 59/100
    50000/50000 [==============================] - 16s 328us/step - loss: 0.9380 - acc: 0.7012 - val_loss: 0.8921 - val_acc: 0.7125
    Epoch 60/100
    50000/50000 [==============================] - 16s 325us/step - loss: 0.9383 - acc: 0.7019 - val_loss: 0.9068 - val_acc: 0.7096
    Epoch 61/100
    50000/50000 [==============================] - 16s 324us/step - loss: 0.9377 - acc: 0.7010 - val_loss: 0.8912 - val_acc: 0.7142
    Epoch 62/100
    50000/50000 [==============================] - 16s 320us/step - loss: 0.9375 - acc: 0.7028 - val_loss: 0.8931 - val_acc: 0.7134
    Epoch 63/100
    50000/50000 [==============================] - 16s 320us/step - loss: 0.9373 - acc: 0.7038 - val_loss: 0.8877 - val_acc: 0.7132
    Epoch 64/100
    50000/50000 [==============================] - 16s 321us/step - loss: 0.9363 - acc: 0.7032 - val_loss: 0.8916 - val_acc: 0.7159
    Epoch 65/100
    50000/50000 [==============================] - 16s 316us/step - loss: 0.9353 - acc: 0.7030 - val_loss: 0.9115 - val_acc: 0.7033
    Epoch 66/100
    50000/50000 [==============================] - 16s 329us/step - loss: 0.9355 - acc: 0.7022 - val_loss: 0.9106 - val_acc: 0.7032
    Epoch 67/100
    50000/50000 [==============================] - 16s 325us/step - loss: 0.9354 - acc: 0.7042 - val_loss: 0.8939 - val_acc: 0.7145
    Epoch 68/100
    50000/50000 [==============================] - 16s 329us/step - loss: 0.9356 - acc: 0.7043 - val_loss: 0.8965 - val_acc: 0.7097
    Epoch 69/100
    50000/50000 [==============================] - 15s 309us/step - loss: 0.9349 - acc: 0.7052 - val_loss: 0.8881 - val_acc: 0.7123
    Epoch 70/100
    50000/50000 [==============================] - 16s 325us/step - loss: 0.9346 - acc: 0.7032 - val_loss: 0.9048 - val_acc: 0.7038
    Epoch 71/100
    50000/50000 [==============================] - 16s 324us/step - loss: 0.9342 - acc: 0.7041 - val_loss: 0.8920 - val_acc: 0.7174
    Epoch 72/100
    50000/50000 [==============================] - 16s 329us/step - loss: 0.9342 - acc: 0.7037 - val_loss: 0.8960 - val_acc: 0.7090
    Epoch 73/100
    50000/50000 [==============================] - 16s 321us/step - loss: 0.9337 - acc: 0.7041 - val_loss: 0.8920 - val_acc: 0.7079
    Epoch 74/100
    50000/50000 [==============================] - 16s 322us/step - loss: 0.9338 - acc: 0.7036 - val_loss: 0.8972 - val_acc: 0.7119
    Epoch 75/100
    50000/50000 [==============================] - 16s 326us/step - loss: 0.9336 - acc: 0.7035 - val_loss: 0.8948 - val_acc: 0.7145
    Epoch 76/100
    50000/50000 [==============================] - 16s 327us/step - loss: 0.9333 - acc: 0.7037 - val_loss: 0.8900 - val_acc: 0.7150
    Epoch 77/100
    50000/50000 [==============================] - 16s 329us/step - loss: 0.9333 - acc: 0.7033 - val_loss: 0.8927 - val_acc: 0.7079
    Epoch 78/100
    50000/50000 [==============================] - 16s 324us/step - loss: 0.9329 - acc: 0.7055 - val_loss: 0.8945 - val_acc: 0.7098
    Epoch 79/100
    50000/50000 [==============================] - 16s 330us/step - loss: 0.9323 - acc: 0.7035 - val_loss: 0.8924 - val_acc: 0.7118
    Epoch 80/100
    50000/50000 [==============================] - 16s 325us/step - loss: 0.9325 - acc: 0.7054 - val_loss: 0.8974 - val_acc: 0.7131
    Epoch 81/100
    50000/50000 [==============================] - 16s 328us/step - loss: 0.9326 - acc: 0.7055 - val_loss: 0.8973 - val_acc: 0.7121
    Epoch 82/100
    50000/50000 [==============================] - 16s 327us/step - loss: 0.9321 - acc: 0.7047 - val_loss: 0.9150 - val_acc: 0.7012
    Epoch 83/100
    50000/50000 [==============================] - 16s 327us/step - loss: 0.9321 - acc: 0.7062 - val_loss: 0.9194 - val_acc: 0.7030
    Epoch 84/100
    50000/50000 [==============================] - 15s 302us/step - loss: 0.9324 - acc: 0.7036 - val_loss: 0.8977 - val_acc: 0.7059
    Epoch 85/100
    50000/50000 [==============================] - 17s 332us/step - loss: 0.9319 - acc: 0.7045 - val_loss: 0.8945 - val_acc: 0.7095
    Epoch 86/100
    50000/50000 [==============================] - 16s 326us/step - loss: 0.9318 - acc: 0.7041 - val_loss: 0.8890 - val_acc: 0.7130
    Epoch 87/100
    50000/50000 [==============================] - 16s 317us/step - loss: 0.9309 - acc: 0.7038 - val_loss: 0.9020 - val_acc: 0.7056
    Epoch 88/100
    50000/50000 [==============================] - 16s 327us/step - loss: 0.9312 - acc: 0.7057 - val_loss: 0.8923 - val_acc: 0.7101
    Epoch 89/100
    50000/50000 [==============================] - 16s 327us/step - loss: 0.9307 - acc: 0.7050 - val_loss: 0.8926 - val_acc: 0.7136
    Epoch 90/100
    50000/50000 [==============================] - 16s 324us/step - loss: 0.9316 - acc: 0.7060 - val_loss: 0.8883 - val_acc: 0.7147
    Epoch 91/100
    50000/50000 [==============================] - 16s 324us/step - loss: 0.9311 - acc: 0.7041 - val_loss: 0.8930 - val_acc: 0.7158
    Epoch 92/100
    50000/50000 [==============================] - 16s 327us/step - loss: 0.9305 - acc: 0.7050 - val_loss: 0.9004 - val_acc: 0.7130
    Epoch 93/100
    50000/50000 [==============================] - 16s 323us/step - loss: 0.9308 - acc: 0.7060 - val_loss: 0.8882 - val_acc: 0.7124
    Epoch 94/100
    50000/50000 [==============================] - 16s 319us/step - loss: 0.9301 - acc: 0.7034 - val_loss: 0.8788 - val_acc: 0.7165
    Epoch 95/100
    50000/50000 [==============================] - 16s 324us/step - loss: 0.9307 - acc: 0.7046 - val_loss: 0.9005 - val_acc: 0.7177
    Epoch 96/100
    50000/50000 [==============================] - 16s 322us/step - loss: 0.9298 - acc: 0.7066 - val_loss: 0.9037 - val_acc: 0.6983
    Epoch 97/100
    50000/50000 [==============================] - 16s 328us/step - loss: 0.9294 - acc: 0.7052 - val_loss: 0.9085 - val_acc: 0.7060
    Epoch 98/100
    50000/50000 [==============================] - 16s 318us/step - loss: 0.9301 - acc: 0.7035 - val_loss: 0.8968 - val_acc: 0.7110
    Epoch 99/100
    50000/50000 [==============================] - 16s 318us/step - loss: 0.9304 - acc: 0.7047 - val_loss: 0.8993 - val_acc: 0.7045
    Epoch 100/100
    50000/50000 [==============================] - 16s 328us/step - loss: 0.9289 - acc: 0.7050 - val_loss: 0.8892 - val_acc: 0.7118
    




    <keras.callbacks.History at 0x1d3bd28e198>



Run below command on the same path of ipynb
    
    tensorboard --logdir=./graph
    
Connect http://localhost:6006

![tensorboard](../assets/img/ML_DL/Tensorboard with Keras/tensorboard.jpg)