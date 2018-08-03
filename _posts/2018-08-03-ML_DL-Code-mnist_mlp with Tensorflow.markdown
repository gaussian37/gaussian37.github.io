---
layout: post
title: mnist_mlp with Tensorflow
date: 2018-08-03 10:30:00
img: ML_DL/tensorflow.png # Add image post (optional)
categories: [ML_DL Code] 
tags: [mlp, mnist, Tensorflow] # add tag
---

+ github : [mnist_mlp with Tensorflow Code](http://nbviewer.jupyter.org/github/gaussian37/Deep-Learning/blob/master/Library/Tensorflow/Multi-layer%20Perceptron/mnist_mlp%20with%20Tensorflow.ipynb)


# mlp with mnist dataset with Tensorflow


```python
import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data
```

```python
tf.set_random_seed(777)
```


```python
mnist = input_data.read_data_sets("./data", one_hot=True)
```
  
Parameters


```python
learning_rate = 0.01
epochs = 15
batch_size = 100
```

Input Placeholder


```python
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
```

Weight & Bias for nn layers

1st layer 


```python
W1 = tf.get_variable(name = "W1", initializer= tf.random_normal([784, 256]))
b1 = tf.get_variable(name = "b1", initializer= tf.random_normal([256]))
a1 = tf.nn.relu(tf.matmul(X, W1) + b1) # (?, 256)
```

2nd layer


```python
W2 = tf.get_variable(name = "W2", initializer= tf.random_normal([256, 256]))
b2 = tf.get_variable(name = "b2", initializer= tf.random_normal([256]))
a2 = tf.nn.relu(tf.matmul(a1, W2) + b2) # (?, 256)
```

last layer


```python
W3 = tf.get_variable(name = "W3", initializer= tf.random_normal([256, 10]))
b3 = tf.get_variable(name = "b3", initializer= tf.random_normal([10]))
hypothesis = tf.matmul(a2, W3) + b3 # (?, 10)
```

define loss & optimizer


```python
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
```

Session run


```python
# For GPU user
# sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth =True)))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(epochs):
        avg_loss = 0
        total_batches = int(mnist.train.num_examples / batch_size)
    
        for i in range(total_batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            feed_dict = {X:batch_x, Y:batch_y}
            _loss, _ = sess.run([loss, optimizer], feed_dict = feed_dict)
            avg_loss += _loss / total_batches

        print("Epoch : {:d} \t loss : {:.4f}".format(epoch, avg_loss))
        
    print("Learning finished")   
    
    # Test model and check accuracy    
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy : {:.2f}".format(sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels})))
    
    # Get one and predict it
    r = random.randint(0, mnist.test.num_examples-1)
    print("Sample : {}, Prediction : {}".format(
        sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)),
        sess.run(tf.argmax(hypothesis, 1), feed_dict={X:mnist.test.images[r:r+1]})))   
```

Epoch : 0 	 loss : 43.5949
Epoch : 1 	 loss : 8.3581
Epoch : 2 	 loss : 4.5213
Epoch : 3 	 loss : 3.0077
Epoch : 4 	 loss : 2.2337
Epoch : 5 	 loss : 1.7791
Epoch : 6 	 loss : 1.7054
Epoch : 7 	 loss : 1.6072
Epoch : 8 	 loss : 1.5188
Epoch : 9 	 loss : 1.2151
Epoch : 10 	 loss : 1.0503
Epoch : 11 	 loss : 0.9019
Epoch : 12 	 loss : 0.8049
Epoch : 13 	 loss : 0.6331
Epoch : 14 	 loss : 0.6727
Learning finished
Accuracy : 0.96
Sample : [8], Prediction : [8]