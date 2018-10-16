---
layout: post
title: mnist_logistic_regression with Tensorflow
date: 2018-07-22 05::00:00
img: ML_DL/tensorflow.png # Add image post (optional)
categories: [machine-learning-code] 
tags: [Logistic Regression, Tensorflow] # add tag
---

+ github : [mnist_logistic_regression with Tensorflow](https://nbviewer.jupyter.org/github/gaussian37/Deep-Learning/blob/master/Library/Tensorflow/Logistic%20Regression/mnist_logistic_regression.ipynb)


# mnist_logistic_regression


```python
import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data
```

### reproducibility


```python
tf.set_random_seed(777)
```


```python
mnist = input_data.read_data_sets("./data", one_hot=True)
```

    Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
    Extracting ./data\train-images-idx3-ubyte.gz
    Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
    Extracting ./data\train-labels-idx1-ubyte.gz
    Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
    Extracting ./data\t10k-images-idx3-ubyte.gz
    Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
    Extracting ./data\t10k-labels-idx1-ubyte.gz
    

### parameters


```python
learning_rate = 0.001
training_epochs = 15
batch_size = 100
```

### Input placeholders


```python
X = tf.placeholder(tf.float32, shape = [None, 28*28], name = "X")
Y = tf.placeholder(tf.float32, shape = [None, 10], name = "Y")
```

### Weight & bias for nn layers


```python
W = tf.get_variable(name = "weights", initializer=tf.random_normal([784, 10]))
b = tf.get_variable(name = "bias", initializer=tf.random_normal([10]))
```

###  hypothesis


```python
hypothesis = tf.matmul(X,W) + b
```

### loss & optimizer


```python
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
```

### Initialize

The first is the `allow_growth` option, which attempts to allocate only as much GPU memory based on runtime allocations: it starts out allocating very little memory, and as Sessions get run and more GPU memory is needed, we extend the GPU memory region needed by the TensorFlow process. Note that we do not release memory, since that can lead to even worse memory fragmentation.


```python
sess = tf.Session(config=tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth = True)))
sess.run(tf.global_variables_initializer())
```

### Train model


```python
for epoch in range(training_epochs):
    avg_loss = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, _loss = sess.run([optimizer, loss], feed_dict = {X:batch_xs, Y:batch_ys})
        avg_loss += _loss / total_batch
     
    print("Epoch : {}, loss : {:.9f}".format(epoch+1, avg_loss))
print("Learning Finished")
```

    Epoch : 1, loss : 0.385825345
    Epoch : 2, loss : 0.376512970
    Epoch : 3, loss : 0.367820557
    Epoch : 4, loss : 0.361098582
    Epoch : 5, loss : 0.354843884
    Epoch : 6, loss : 0.348474874
    Epoch : 7, loss : 0.343070328
    Epoch : 8, loss : 0.337118507
    Epoch : 9, loss : 0.332395571
    Epoch : 10, loss : 0.328015425
    Epoch : 11, loss : 0.323890320
    Epoch : 12, loss : 0.319517709
    Epoch : 13, loss : 0.315231980
    Epoch : 14, loss : 0.312611983
    Epoch : 15, loss : 0.308055050
    Learning Finished
    

### Test model and check accuracy


```python
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accruracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy : ", sess.run(accruracy, feed_dict = {X:mnist.test.images, Y:mnist.test.labels}))
```

    Accuracy :  0.9153
    


```python
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))
```

    Label:  [7]
    Prediction:  [7]