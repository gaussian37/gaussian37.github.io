---
layout: post
title: Early stopping with Keras
date: 2018-07-25 10:30:00
img: ML_DL/keras.png # Add image post (optional)
categories: [ML_DL Code] 
tags: [callback, Early stopping, Keras] # add tag
---

+ github : [Early Stopping with Keras Code](https://nbviewer.jupyter.org/github/gaussian37/Deep-Learning/blob/master/Library/Keras/Keras%20Reference/Early%20Stopping%20with%20Keras.ipynb)


# Early Stopping with Keras

In order to early stop the learning, We can use 'EarlyStopping()' function. This is the callback function and we can use it when the learning algorithm can not improve the learning status. 

Callback function means that when you call a function, callback function calls specific function which I designated.

In kears, EarlyStopping() callback function is called in fit() function.

EarlyStopping() callback function has many option. Let's check those out!

+ monitor
    - Items to observe. **"val_loss", "val_acc"**
  
+ min_delta
    - It indicates the **minimum amount of change** to be determined to be improving. If the amount of changing is less than min_delta, it is judged that there is no improvement.

+ patience
    - Specify **how long to wait** the non-improvement epoch and not to stop immediately even though there is no improvement. If you set this value as 10, learning ends when consecutive 10 times no improvement happens.

### Import package


```python
from keras.utils import np_utils
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from keras.regularizers import l2
np.random.seed(3)
```

    Using TensorFlow backend.
    

### Dataset


```python
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
```

### split the dataset


```python
x_val = x_train[50000:]
y_val = y_train[50000:]
x_train = x_train[:50000]
y_train = y_train[:50000]
```

### preprocessing


```python
x_train = x_train.reshape(50000, 784).astype("float32") / 255.0
x_val = x_val.reshape(10000, 784).astype("float32") / 255.0
x_test = x_test.reshape(10000, 784).astype("float32") / 255.0
```

### One-hot encoding process


```python
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)
y_test = np_utils.to_categorical(y_test)
```

### Modeling


```python
model = Sequential()
model.add(Dense(units = 50, input_dim = 28*28, activation = "relu", W_regularizer=l2(0.01)))
model.add(Dense(units = 30, activation= "relu", W_regularizer=l2(0.01)))
model.add(Dense(units= 10, activation="relu",W_regularizer=l2(0.01)))
model.add(Dense(units= 10, activation="softmax",W_regularizer=l2(0.01)))

model.compile(loss = "categorical_crossentropy", optimizer= "adam", metrics=["accuracy"])
```

### Learning


```python
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=20)
hist = model.fit(x_train, y_train, epochs=1000, batch_size=32, 
                 validation_data=(x_val, y_val), callbacks=[early_stopping])
```

50000/50000 [==============================] - 20s 408us/step - loss: 1.1935 - acc: 0.7163 - val_loss: 1.1089 - val_acc: 0.7328
Train on 50000 samples, validate on 10000 samples
Epoch 1/1000
Epoch 2/1000
50000/50000 [==============================] - 19s 378us/step - loss: 0.9925 - acc: 0.7760 - val_loss: 0.9935 - val_acc: 0.7491
Epoch 3/1000
50000/50000 [==============================] - 18s 368us/step - loss: 0.9630 - acc: 0.7832 - val_loss: 0.9464 - val_acc: 0.7857
Epoch 4/1000
50000/50000 [==============================] - 19s 381us/step - loss: 0.9452 - acc: 0.7895 - val_loss: 0.9294 - val_acc: 0.7882
Epoch 5/1000
50000/50000 [==============================] - 19s 380us/step - loss: 0.9331 - acc: 0.7936 - val_loss: 0.9184 - val_acc: 0.7937
Epoch 6/1000
50000/50000 [==============================] - 19s 373us/step - loss: 0.9272 - acc: 0.7940 - val_loss: 0.9477 - val_acc: 0.7880
Epoch 7/1000
50000/50000 [==============================] - 19s 374us/step - loss: 0.9215 - acc: 0.7966 - val_loss: 0.9514 - val_acc: 0.7844
Epoch 8/1000
50000/50000 [==============================] - 19s 371us/step - loss: 0.9176 - acc: 0.7959 - val_loss: 0.9334 - val_acc: 0.7860
Epoch 9/1000
50000/50000 [==============================] - 18s 369us/step - loss: 0.9144 - acc: 0.7979 - val_loss: 0.9254 - val_acc: 0.7841
Epoch 10/1000
50000/50000 [==============================] - 19s 372us/step - loss: 0.9141 - acc: 0.7979 - val_loss: 0.9010 - val_acc: 0.8018
Epoch 11/1000
50000/50000 [==============================] - 18s 370us/step - loss: 0.9135 - acc: 0.7950 - val_loss: 0.9080 - val_acc: 0.8000
Epoch 12/1000
50000/50000 [==============================] - 18s 369us/step - loss: 0.9127 - acc: 0.7977 - val_loss: 0.9341 - val_acc: 0.7877
Epoch 13/1000
50000/50000 [==============================] - 18s 366us/step - loss: 0.9108 - acc: 0.7972 - val_loss: 0.9427 - val_acc: 0.7824
Epoch 14/1000
50000/50000 [==============================] - 19s 374us/step - loss: 0.9102 - acc: 0.7975 - val_loss: 0.9332 - val_acc: 0.7862
Epoch 15/1000
50000/50000 [==============================] - 19s 377us/step - loss: 0.9117 - acc: 0.7978 - val_loss: 0.9249 - val_acc: 0.7971
Epoch 16/1000
50000/50000 [==============================] - 19s 371us/step - loss: 0.9091 - acc: 0.7983 - val_loss: 0.9147 - val_acc: 0.7972
Epoch 17/1000
50000/50000 [==============================] - 19s 374us/step - loss: 0.9098 - acc: 0.7963 - val_loss: 0.9052 - val_acc: 0.7969
Epoch 18/1000
50000/50000 [==============================] - 19s 373us/step - loss: 0.9082 - acc: 0.7975 - val_loss: 0.8882 - val_acc: 0.8037
Epoch 19/1000
50000/50000 [==============================] - 18s 369us/step - loss: 0.9069 - acc: 0.7985 - val_loss: 0.9150 - val_acc: 0.7958
Epoch 20/1000
50000/50000 [==============================] - 19s 373us/step - loss: 0.9061 - acc: 0.7982 - val_loss: 0.9164 - val_acc: 0.7956
Epoch 21/1000
50000/50000 [==============================] - 18s 368us/step - loss: 0.9072 - acc: 0.7966 - val_loss: 0.9089 - val_acc: 0.7934
Epoch 22/1000
50000/50000 [==============================] - 18s 368us/step - loss: 0.9050 - acc: 0.7981 - val_loss: 0.9000 - val_acc: 0.7921
Epoch 23/1000
50000/50000 [==============================] - 18s 362us/step - loss: 0.9043 - acc: 0.7964 - val_loss: 0.9095 - val_acc: 0.7937
Epoch 24/1000
50000/50000 [==============================] - 18s 366us/step - loss: 0.9057 - acc: 0.7989 - val_loss: 0.9293 - val_acc: 0.7901
Epoch 25/1000
50000/50000 [==============================] - 19s 371us/step - loss: 0.9050 - acc: 0.7977 - val_loss: 0.8902 - val_acc: 0.8019
Epoch 26/1000
50000/50000 [==============================] - 19s 372us/step - loss: 0.9046 - acc: 0.7976 - val_loss: 0.9003 - val_acc: 0.7956
Epoch 27/1000
50000/50000 [==============================] - 19s 374us/step - loss: 0.9055 - acc: 0.7982 - val_loss: 0.9425 - val_acc: 0.7737
Epoch 28/1000
50000/50000 [==============================] - 18s 364us/step - loss: 0.9013 - acc: 0.8007 - val_loss: 0.9158 - val_acc: 0.7932
Epoch 29/1000
50000/50000 [==============================] - 19s 375us/step - loss: 0.9041 - acc: 0.7987 - val_loss: 0.8907 - val_acc: 0.7972
Epoch 30/1000
50000/50000 [==============================] - 19s 372us/step - loss: 0.9047 - acc: 0.7984 - val_loss: 0.8933 - val_acc: 0.7985
Epoch 31/1000
50000/50000 [==============================] - 19s 374us/step - loss: 0.9023 - acc: 0.7985 - val_loss: 0.9103 - val_acc: 0.7887
Epoch 32/1000
50000/50000 [==============================] - 19s 376us/step - loss: 0.9036 - acc: 0.7992 - val_loss: 0.8995 - val_acc: 0.7974
Epoch 33/1000
50000/50000 [==============================] - 19s 372us/step - loss: 0.9031 - acc: 0.7981 - val_loss: 0.9398 - val_acc: 0.7820
Epoch 34/1000
50000/50000 [==============================] - 18s 368us/step - loss: 0.9039 - acc: 0.8004 - val_loss: 0.9014 - val_acc: 0.7946
Epoch 35/1000
50000/50000 [==============================] - 18s 366us/step - loss: 0.9045 - acc: 0.7986 - val_loss: 0.8989 - val_acc: 0.7923
Epoch 36/1000
50000/50000 [==============================] - 18s 360us/step - loss: 0.9013 - acc: 0.7989 - val_loss: 0.8973 - val_acc: 0.7937
Epoch 37/1000
50000/50000 [==============================] - 19s 371us/step - loss: 0.9033 - acc: 0.7973 - val_loss: 0.9232 - val_acc: 0.7875
Epoch 38/1000
50000/50000 [==============================] - 19s 375us/step - loss: 0.9020 - acc: 0.7991 - val_loss: 0.9169 - val_acc: 0.7834
  

### Display


```python
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history["loss"], "y", label = "train_loss")
loss_ax.plot(hist.history["val_loss"], "r", label = "val_loss")

acc_ax.plot(hist.history["acc"], "b", label="train_acc")
acc_ax.plot(hist.history["val_acc"], "g", label="val_acc")

loss_ax.set_ylabel("loss")
acc_ax.set_ylabel("accuracy")

loss_ax.legend(loc = "upper left")
acc_ax.legend(loc = "lower left")

plt.show()
```


![png](../assets/img/ML_DL/Early Stopping with Keras/Early Stopping with Keras_18_0.png)


```python
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
print("")
print("loss : {}".format(loss_and_metrics[0]))
print("accuracy : {}".format(loss_and_metrics[1]))
```

    10000/10000 [==============================] - 0s 44us/step
    
    loss : 0.929823195362091
    accuracy : 0.7799
    