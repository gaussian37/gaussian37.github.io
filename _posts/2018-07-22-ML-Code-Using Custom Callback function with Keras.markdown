---
layout: post
title: Using Custom Callback function with Keras
date: 2018-07-22 05:01:00
img: ML_DL/keras.png # Add image post (optional)
categories: [ML_DL Code] 
tags: [callback, keras] # add tag
---

+ github : [Using Custom Callback function with Keras](https://nbviewer.jupyter.org/github/gaussian37/Deep-Learning/blob/master/Library/Keras/Keras%20Reference/Using%20Custom%20Callback%20function%20with%20Keras.ipynb)


# Using Custom Callback function with Keras

You can use history callback function or tensorboard in basic model monitoring. But, if you monitor another model like Recurrent Neural Network, you would not monitor properly because RNN call **fit function**  many times. Let me give you a example.


```python
for epoch in range(1000):
    print("Epochs : " + str(epoch))
    hist = model.fit(x_train, y_train, epochs = 1, batch_size = 1, verbose = 2, shuffle=False)
    model.reset_states()
```

In above code, You can not observe the trend by every epoch because in every epoch, new history object replaces existing object. Accordingly, You need to `define custom callback function in order to maintain the existing learning state.`
Let me give you a example.


```python
import keras
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

np.random.seed(3)

```

Using TensorFlow backend.    
    

### Define custom callback function


```python
class CustomHistory(keras.callbacks.Callback):
    def init(self):
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        
    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.train_acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
```

### Dataset


```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)


x_val = x_train[50000:]
y_val = y_train[50000:]
x_train = x_train[:50000]
y_train = y_train[:50000]
```

### preprocessing dataset


```python
x_train = x_train.reshape(50000, 784).astype("float32") / 255.0
x_val = x_val.reshape(10000, 784).astype("float32") / 255.0
x_test = x_test.reshape(10000, 784).astype("float32") / 255.0
```

### Randomize dataset


```python
train_rand_idx = np.arange(0, x_train.shape[0])
val_rand_idx = np.arange(0, x_val.shape[0])
np.random.shuffle(train_rand_idx)
np.random.shuffle(val_rand_idx)

x_train = x_train[train_rand_idx]
y_train = y_train[train_rand_idx]
x_val = x_val[val_rand_idx]
y_val = y_val[val_rand_idx]
```

### One-hot encoding the labels


```python
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)
y_test = np_utils.to_categorical(y_test)
```

### Modeling


```python
model = Sequential()
model.add(Dense(units = 2, input_dim = 28*28, activation = "relu"))
model.add(Dense(units = 10, activation = "softmax"))
```

### Compile


```python
model.compile(loss = "categorical_crossentropy", optimizer = "sgd", metrics = ["accuracy"])
```

### Learning

Basically, you can set the epoch option in model.fit (ex. model.fit(epochs = 1000)). But in this example, we set the epoch option as 1 and use the for-loop in order to use custom callback function.


```python
custom_hist = CustomHistory()
custom_hist.init()

for epoch in range(100):
    print("Epoch : {}".format(epoch))
    model.fit(x_train, y_train, batch_size=10, epochs=1, validation_data=(x_val, y_val), callbacks=[custom_hist])
```

output

    Epoch : 0
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 18s 356us/step - loss: 1.6571 - acc: 0.3595 - val_loss: 1.4322 - val_acc: 0.4336
    Epoch : 1
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 333us/step - loss: 1.3770 - acc: 0.4512 - val_loss: 1.3054 - val_acc: 0.4753
    Epoch : 2
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 341us/step - loss: 1.2902 - acc: 0.4804 - val_loss: 1.2532 - val_acc: 0.4800
    Epoch : 3
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 340us/step - loss: 1.2512 - acc: 0.4958 - val_loss: 1.2210 - val_acc: 0.5155
    Epoch : 4
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 341us/step - loss: 1.2272 - acc: 0.5108 - val_loss: 1.2011 - val_acc: 0.5208
    Epoch : 5
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 333us/step - loss: 1.2109 - acc: 0.5211 - val_loss: 1.1898 - val_acc: 0.5331
    Epoch : 6
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 336us/step - loss: 1.1984 - acc: 0.5286 - val_loss: 1.1773 - val_acc: 0.5255
    Epoch : 7
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 333us/step - loss: 1.1879 - acc: 0.5352 - val_loss: 1.1646 - val_acc: 0.5427
    Epoch : 8
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 337us/step - loss: 1.1785 - acc: 0.5367 - val_loss: 1.1608 - val_acc: 0.5495
    Epoch : 9
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 339us/step - loss: 1.1690 - acc: 0.5417 - val_loss: 1.1520 - val_acc: 0.5428
    Epoch : 10
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 337us/step - loss: 1.1629 - acc: 0.5435 - val_loss: 1.1432 - val_acc: 0.5444
    Epoch : 11
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 16s 321us/step - loss: 1.1579 - acc: 0.5486 - val_loss: 1.1371 - val_acc: 0.5612
    Epoch : 12
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 337us/step - loss: 1.1533 - acc: 0.5518 - val_loss: 1.1298 - val_acc: 0.5601
    Epoch : 13
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 337us/step - loss: 1.1502 - acc: 0.5558 - val_loss: 1.1327 - val_acc: 0.5557
    Epoch : 14
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 16s 321us/step - loss: 1.1464 - acc: 0.5580 - val_loss: 1.1321 - val_acc: 0.5570
    Epoch : 15
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 339us/step - loss: 1.1442 - acc: 0.5591 - val_loss: 1.1235 - val_acc: 0.5601
    Epoch : 16
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 16s 329us/step - loss: 1.1426 - acc: 0.5599 - val_loss: 1.1267 - val_acc: 0.5700
    Epoch : 17
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 333us/step - loss: 1.1409 - acc: 0.5626 - val_loss: 1.1187 - val_acc: 0.5702
    Epoch : 18
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 337us/step - loss: 1.1387 - acc: 0.5636 - val_loss: 1.1202 - val_acc: 0.5729
    Epoch : 19
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 334us/step - loss: 1.1372 - acc: 0.5637 - val_loss: 1.1273 - val_acc: 0.5598
    Epoch : 20
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 336us/step - loss: 1.1356 - acc: 0.5628 - val_loss: 1.1346 - val_acc: 0.5654
    Epoch : 21
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 337us/step - loss: 1.1350 - acc: 0.5661 - val_loss: 1.1168 - val_acc: 0.5691
    Epoch : 22
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 337us/step - loss: 1.1331 - acc: 0.5660 - val_loss: 1.1166 - val_acc: 0.5670
    Epoch : 23
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 16s 328us/step - loss: 1.1309 - acc: 0.5670 - val_loss: 1.1090 - val_acc: 0.5752
    Epoch : 24
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 16s 327us/step - loss: 1.1304 - acc: 0.5680 - val_loss: 1.1055 - val_acc: 0.5771
    Epoch : 25
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 334us/step - loss: 1.1293 - acc: 0.5716 - val_loss: 1.1124 - val_acc: 0.5888
    Epoch : 26
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 337us/step - loss: 1.1271 - acc: 0.5732 - val_loss: 1.0965 - val_acc: 0.5913
    Epoch : 27
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 336us/step - loss: 1.1214 - acc: 0.5907 - val_loss: 1.0904 - val_acc: 0.6045
    Epoch : 28
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 335us/step - loss: 1.1118 - acc: 0.6022 - val_loss: 1.0744 - val_acc: 0.6253
    Epoch : 29
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 16s 324us/step - loss: 1.1043 - acc: 0.6071 - val_loss: 1.0732 - val_acc: 0.6256
    Epoch : 30
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 16s 328us/step - loss: 1.0999 - acc: 0.6114 - val_loss: 1.0706 - val_acc: 0.6260
    Epoch : 31
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 336us/step - loss: 1.0965 - acc: 0.6129 - val_loss: 1.0711 - val_acc: 0.6221
    Epoch : 32
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 338us/step - loss: 1.0938 - acc: 0.6126 - val_loss: 1.0617 - val_acc: 0.6443
    Epoch : 33
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 335us/step - loss: 1.0922 - acc: 0.6152 - val_loss: 1.0614 - val_acc: 0.6338
    Epoch : 34
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 333us/step - loss: 1.0904 - acc: 0.6130 - val_loss: 1.0578 - val_acc: 0.6294
    Epoch : 35
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 16s 330us/step - loss: 1.0891 - acc: 0.6160 - val_loss: 1.0613 - val_acc: 0.6228
    Epoch : 36
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 16s 327us/step - loss: 1.0878 - acc: 0.6147 - val_loss: 1.0707 - val_acc: 0.6230
    Epoch : 37
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 16s 330us/step - loss: 1.0868 - acc: 0.6135 - val_loss: 1.0569 - val_acc: 0.6319
    Epoch : 38
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 335us/step - loss: 1.0853 - acc: 0.6147 - val_loss: 1.0479 - val_acc: 0.6326
    Epoch : 39
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 338us/step - loss: 1.0839 - acc: 0.6173 - val_loss: 1.0628 - val_acc: 0.6218
    Epoch : 40
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 340us/step - loss: 1.0828 - acc: 0.6139 - val_loss: 1.0508 - val_acc: 0.6278
    Epoch : 41
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 334us/step - loss: 1.0817 - acc: 0.6161 - val_loss: 1.0510 - val_acc: 0.6324
    Epoch : 42
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 16s 328us/step - loss: 1.0810 - acc: 0.6152 - val_loss: 1.0524 - val_acc: 0.6250
    Epoch : 43
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 334us/step - loss: 1.0805 - acc: 0.6153 - val_loss: 1.0531 - val_acc: 0.6210
    Epoch : 44
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 16s 330us/step - loss: 1.0790 - acc: 0.6187 - val_loss: 1.0538 - val_acc: 0.6205
    Epoch : 45
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 16s 328us/step - loss: 1.0786 - acc: 0.6169 - val_loss: 1.0678 - val_acc: 0.6081
    Epoch : 46
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 337us/step - loss: 1.0771 - acc: 0.6166 - val_loss: 1.0447 - val_acc: 0.6251
    Epoch : 47
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 338us/step - loss: 1.0768 - acc: 0.6173 - val_loss: 1.0547 - val_acc: 0.6239
    Epoch : 48
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 16s 312us/step - loss: 1.0755 - acc: 0.6176 - val_loss: 1.0498 - val_acc: 0.6254
    Epoch : 49
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 332us/step - loss: 1.0748 - acc: 0.6154 - val_loss: 1.0620 - val_acc: 0.6143
    Epoch : 50
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 336us/step - loss: 1.0746 - acc: 0.6172 - val_loss: 1.0437 - val_acc: 0.6340
    Epoch : 51
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 335us/step - loss: 1.0737 - acc: 0.6186 - val_loss: 1.0483 - val_acc: 0.6281
    Epoch : 52
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 332us/step - loss: 1.0730 - acc: 0.6184 - val_loss: 1.0449 - val_acc: 0.6273
    Epoch : 53
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 332us/step - loss: 1.0730 - acc: 0.6184 - val_loss: 1.0446 - val_acc: 0.6209
    Epoch : 54
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 341us/step - loss: 1.0718 - acc: 0.6183 - val_loss: 1.0444 - val_acc: 0.6414
    Epoch : 55
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 330us/step - loss: 1.0713 - acc: 0.6194 - val_loss: 1.0584 - val_acc: 0.6139
    Epoch : 56
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 335us/step - loss: 1.0710 - acc: 0.6195 - val_loss: 1.0438 - val_acc: 0.6326
    Epoch : 57
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 337us/step - loss: 1.0698 - acc: 0.6204 - val_loss: 1.0426 - val_acc: 0.6230
    Epoch : 58
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 335us/step - loss: 1.0696 - acc: 0.6206 - val_loss: 1.0572 - val_acc: 0.6142
    Epoch : 59
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 336us/step - loss: 1.0702 - acc: 0.6176 - val_loss: 1.0397 - val_acc: 0.6325
    Epoch : 60
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 331us/step - loss: 1.0687 - acc: 0.6198 - val_loss: 1.0413 - val_acc: 0.6375
    Epoch : 61
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 334us/step - loss: 1.0690 - acc: 0.6189 - val_loss: 1.0428 - val_acc: 0.6220
    Epoch : 62
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 336us/step - loss: 1.0685 - acc: 0.6177 - val_loss: 1.0392 - val_acc: 0.6277
    Epoch : 63
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 336us/step - loss: 1.0681 - acc: 0.6181 - val_loss: 1.0340 - val_acc: 0.6382
    Epoch : 64
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 333us/step - loss: 1.0675 - acc: 0.6199 - val_loss: 1.0395 - val_acc: 0.6233
    Epoch : 65
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 334us/step - loss: 1.0672 - acc: 0.6208 - val_loss: 1.0368 - val_acc: 0.6169
    Epoch : 66
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 334us/step - loss: 1.0672 - acc: 0.6198 - val_loss: 1.0460 - val_acc: 0.6259
    Epoch : 67
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 16s 311us/step - loss: 1.0665 - acc: 0.6204 - val_loss: 1.0340 - val_acc: 0.6425
    Epoch : 68
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 332us/step - loss: 1.0668 - acc: 0.6199 - val_loss: 1.0432 - val_acc: 0.6250
    Epoch : 69
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 16s 327us/step - loss: 1.0666 - acc: 0.6194 - val_loss: 1.0384 - val_acc: 0.6264
    Epoch : 70
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 333us/step - loss: 1.0653 - acc: 0.6183 - val_loss: 1.0482 - val_acc: 0.6227
    Epoch : 71
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 332us/step - loss: 1.0648 - acc: 0.6195 - val_loss: 1.0689 - val_acc: 0.6123
    Epoch : 72
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 16s 312us/step - loss: 1.0650 - acc: 0.6188 - val_loss: 1.0553 - val_acc: 0.6124
    Epoch : 73
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 16s 320us/step - loss: 1.0648 - acc: 0.6194 - val_loss: 1.0650 - val_acc: 0.6157
    Epoch : 74
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 332us/step - loss: 1.0642 - acc: 0.6212 - val_loss: 1.0424 - val_acc: 0.6286
    Epoch : 75
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 336us/step - loss: 1.0643 - acc: 0.6213 - val_loss: 1.0417 - val_acc: 0.6438
    Epoch : 76
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 337us/step - loss: 1.0640 - acc: 0.6191 - val_loss: 1.0403 - val_acc: 0.6370
    Epoch : 77
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 335us/step - loss: 1.0638 - acc: 0.6198 - val_loss: 1.0384 - val_acc: 0.6314
    Epoch : 78
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 331us/step - loss: 1.0633 - acc: 0.6216 - val_loss: 1.0417 - val_acc: 0.6293
    Epoch : 79
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 332us/step - loss: 1.0637 - acc: 0.6194 - val_loss: 1.0374 - val_acc: 0.6330
    Epoch : 80
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 16s 325us/step - loss: 1.0630 - acc: 0.6200 - val_loss: 1.0386 - val_acc: 0.6242
    Epoch : 81
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 338us/step - loss: 1.0635 - acc: 0.6208 - val_loss: 1.0404 - val_acc: 0.6279
    Epoch : 82
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 334us/step - loss: 1.0624 - acc: 0.6207 - val_loss: 1.0373 - val_acc: 0.6245
    Epoch : 83
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 334us/step - loss: 1.0623 - acc: 0.6208 - val_loss: 1.0338 - val_acc: 0.6328
    Epoch : 84
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 335us/step - loss: 1.0625 - acc: 0.6222 - val_loss: 1.0397 - val_acc: 0.6194
    Epoch : 85
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 16s 325us/step - loss: 1.0625 - acc: 0.6189 - val_loss: 1.0303 - val_acc: 0.6368
    Epoch : 86
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 16s 320us/step - loss: 1.0626 - acc: 0.6208 - val_loss: 1.0335 - val_acc: 0.6363
    Epoch : 87
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 334us/step - loss: 1.0618 - acc: 0.6199 - val_loss: 1.0343 - val_acc: 0.6401
    Epoch : 88
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 334us/step - loss: 1.0610 - acc: 0.6225 - val_loss: 1.0378 - val_acc: 0.6223
    Epoch : 89
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 334us/step - loss: 1.0608 - acc: 0.6227 - val_loss: 1.0452 - val_acc: 0.6214
    Epoch : 90
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 333us/step - loss: 1.0619 - acc: 0.6203 - val_loss: 1.0379 - val_acc: 0.6225
    Epoch : 91
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 332us/step - loss: 1.0608 - acc: 0.6214 - val_loss: 1.0367 - val_acc: 0.6404
    Epoch : 92
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 338us/step - loss: 1.0608 - acc: 0.6223 - val_loss: 1.0330 - val_acc: 0.6366
    Epoch : 93
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 16s 328us/step - loss: 1.0607 - acc: 0.6229 - val_loss: 1.0444 - val_acc: 0.6356
    Epoch : 94
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 336us/step - loss: 1.0605 - acc: 0.6220 - val_loss: 1.0324 - val_acc: 0.6407
    Epoch : 95
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 335us/step - loss: 1.0609 - acc: 0.6209 - val_loss: 1.0405 - val_acc: 0.6306
    Epoch : 96
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 332us/step - loss: 1.0603 - acc: 0.6216 - val_loss: 1.0351 - val_acc: 0.6411
    Epoch : 97
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 16s 325us/step - loss: 1.0602 - acc: 0.6219 - val_loss: 1.0411 - val_acc: 0.6287
    Epoch : 98
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 334us/step - loss: 1.0601 - acc: 0.6211 - val_loss: 1.0338 - val_acc: 0.6238
    Epoch : 99
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/1
    50000/50000 [==============================] - 17s 337us/step - loss: 1.0597 - acc: 0.6202 - val_loss: 1.0356 - val_acc: 0.6299
    

Display the results

```python
import matplotlib.pyplot as plt
%matplotlib inline
fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(custom_hist.train_loss, 'y', label = 'train loss')
loss_ax.plot(custom_hist.val_loss, 'r', label = 'val loss')

acc_ax.plot(custom_hist.train_acc, 'b', label = 'train acc')
acc_ax.plot(custom_hist.val_acc, 'g', label = 'val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuaracy')

loss_ax.legend(loc = 'upper left')
acc_ax.legend(loc = 'lower left')
```

![png](../assets/img/ML_DL/callback function with keras/Using Custom Callback function with Keras_22_1.png)

