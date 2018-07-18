---
layout: post
title: Linear Regression from Text with Tensorflow
date: 2018-07-17 05:09:00
img: ML/tensorflow.png # Add image post (optional)
categories: [ML_DL Code] 
tags: [Linear Regression, Tensorflow] # add tag
---

+ github : [Linear Regression from Text with Tensorflow](https://nbviewer.jupyter.org/github/gaussian37/Deep-Learning/blob/master/Library/Tensorflow/Linear%20Regression/Linear%20Regression%20from%20Text%20with%20Tensorflow.ipynb)
+ data : [data](https://github.com/gaussian37/Deep-Learning/blob/master/Library/Tensorflow/Linear%20Regression/data/birth_life_2010.txt)

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
```
<br>

Read the data file
Linear Regression from Text with Tensorflow.ipynb <br>
data <br>
└── birth_life_2010.txt <br>

+ birth_life_2010.txt

| Country | Birth rate | Life expectancy |
|---------|------------|-----------------|
| Vietnam | 1.822      | 74.828243902    |
| Vanuatu | 3.869      | 70.819487805    |
| ...     | ...        | ...             |



```python
DATA_FILE = "./data/birth_life_2010.txt"
```


```python
def read_birth_file_data(filename):
    """
    Read in birth_life_2010.txt and return:
    data in the form of NumPy array
    n_samples: number of samples
    """        
    "readline from 1 row (except 0 row : category)"
    text = open(filename, 'r').readlines()[1:]   
    "Split each line with '\t'"
    data = [line[:-1].split('\t') for line in text]
    "Select the column 1 of birth"
    births = [float(line[1]) for line in data]
    "Select the column 2 of lifes"
    lifes = [float(line[2]) for line in data]
    "Zip birth & lifes"
    data = list(zip(births, lifes))
    "The number of samples"
    n_samples = len(data)
    "Transform data type from list to np.array"
    data = np.asarray(data, dtype=np.float32)
    
    return data, n_samples
```

#### Step 1 : Read in data from the .txt file


```python
data, n_samples = read_birth_file_data(DATA_FILE)
```

#### Step 2: Create placeholders for X (birth rate) and Y (life expectancy)


```python
X = tf.placeholder(tf.float32, name = "X")
Y = tf.placeholder(tf.float32, name = "Y")
```

#### Step 3 : create weight and bias, initialized to 0


```python
w = tf.get_variable("weights", initializer=tf.constant(0.0))
b = tf.get_variable("bias", initializer=tf.constant(0.0))
```

#### Step 4 : build model to predict Y


```python
hypothesis = w * X + b
```

#### Step 5 : use the squared error as the loss function


```python
loss = tf.reduce_mean(tf.square(Y - hypothesis, name = 'loss'))
```

#### Step 6 : Using gradient descent with learning rate of 0.001 to minimize loss


```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
```

#### Stetp 7 : initialize the necessary variables, in this case, w and b
#### Stetp 8 : train the model for 100 epochs
#### Stetp 9 : output the values of w and b

see below code.


```python
writer = tf.summary.FileWriter("./graphs/linear_Regression_Birth_Life", tf.get_default_graph())
```


```python
with tf.Session() as sess:
    # Stetp 7 : initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())
    
    # Stetp 8 : train the model for 100 epochs
    for i in range(100):
        total_loss = 0
        for x, y in data:
            # Session execute optimizer and fetch values of loss
            _, _loss = sess.run([optimizer, loss], feed_dict = {X:x, Y:y})
            total_loss += _loss
        print("Epoch {0} : {1}".format(i, total_loss / n_samples))
    
    # close the writer when you're done using it
    writer.close()
    
    # Step 9 : output the values of w and b
    w_out, b_out = sess.run([w, b])   
```
<br><br>

    Epoch 0 : 1661.8637834631543
    Epoch 1 : 956.3224148609137
    Epoch 2 : 844.6737023980994
    Epoch 3 : 750.7312486011339
    Epoch 4 : 667.6598341012079
    Epoch 5 : 594.1417715627896
    Epoch 6 : 529.07878103068
    Epoch 7 : 471.5004191489204
    Epoch 8 : 420.5458626462441
    Epoch 9 : 375.45530721966765
    Epoch 10 : 335.5543025185697
    Epoch 11 : 300.24629857978107
    Epoch 12 : 269.00376475843336
    Epoch 13 : 241.35957466852116
    Epoch 14 : 216.90039135300015
    Epoch 15 : 195.25972298129324
    Epoch 16 : 176.1137693605349
    Epoch 17 : 159.17551693441837
    Epoch 18 : 144.1907111125557
    Epoch 19 : 130.93503488078713
    Epoch 20 : 119.20935661137888
    Epoch 21 : 108.8379309807855
    Epoch 22 : 99.66466760624593
    Epoch 23 : 91.55177013029001
    Epoch 24 : 84.37664046781751
    Epoch 25 : 78.03217824997724
    Epoch 26 : 72.42182927812989
    Epoch 27 : 67.46136239485718
    Epoch 28 : 63.07566952367442
    Epoch 29 : 59.19874146522856
    Epoch 30 : 55.77168446383194
    Epoch 31 : 52.74269822355127
    Epoch 32 : 50.065632780875376
    Epoch 33 : 47.70006421631674
    Epoch 34 : 45.61017902122909
    Epoch 35 : 43.76379750625255
    Epoch 36 : 42.13259221098116
    Epoch 37 : 40.69221939330516
    Epoch 38 : 39.420219863367905
    Epoch 39 : 38.297008645340895
    Epoch 40 : 37.305591759538146
    Epoch 41 : 36.43066341609841
    Epoch 42 : 35.658453942681234
    Epoch 43 : 34.97724816803575
    Epoch 44 : 34.37655378567349
    Epoch 45 : 33.84671358035044
    Epoch 46 : 33.379665882282545
    Epoch 47 : 32.96800991297258
    Epoch 48 : 32.60548541990942
    Epoch 49 : 32.28618434173986
    Epoch 50 : 32.004961317298495
    Epoch 51 : 31.757531331044525
    Epoch 52 : 31.53978877073019
    Epoch 53 : 31.348356819100445
    Epoch 54 : 31.180119247269193
    Epoch 55 : 31.03225782010038
    Epoch 56 : 30.902462910201574
    Epoch 57 : 30.78859985760776
    Epoch 58 : 30.688725355066556
    Epoch 59 : 30.60122861903357
    Epoch 60 : 30.524590178362192
    Epoch 61 : 30.457532704476954
    Epoch 62 : 30.398967422668726
    Epoch 63 : 30.34777825418737
    Epoch 64 : 30.303121465726413
    Epoch 65 : 30.26424930739051
    Epoch 66 : 30.230392129550456
    Epoch 67 : 30.200964921590334
    Epoch 68 : 30.175501555469697
    Epoch 69 : 30.153343991707324
    Epoch 70 : 30.134226098457216
    Epoch 71 : 30.117758308603477
    Epoch 72 : 30.103543774372174
    Epoch 73 : 30.09139442229674
    Epoch 74 : 30.0809388476427
    Epoch 75 : 30.07208499982095
    Epoch 76 : 30.06452690966084
    Epoch 77 : 30.058150938555205
    Epoch 78 : 30.05278219980139
    Epoch 79 : 30.04828310612785
    Epoch 80 : 30.04458791257593
    Epoch 81 : 30.041550708114855
    Epoch 82 : 30.039046437352113
    Epoch 83 : 30.03704103724602
    Epoch 84 : 30.03545715799831
    Epoch 85 : 30.034288759106282
    Epoch 86 : 30.03338805212261
    Epoch 87 : 30.032769865304076
    Epoch 88 : 30.032386838833535
    Epoch 89 : 30.032150670733166
    Epoch 90 : 30.032092865493677
    Epoch 91 : 30.032186730024037
    Epoch 92 : 30.03240725137661
    Epoch 93 : 30.032643962397827
    Epoch 94 : 30.033039376884087
    Epoch 95 : 30.033435566514413
    Epoch 96 : 30.033922631802085
    Epoch 97 : 30.03442924663878
    Epoch 98 : 30.0349335548615
    Epoch 99 : 30.03552558278714
    
<br><br>    


```python
# plot the results
plt.plot(data[:,0], data[:,1], "bo", label = "Real Data")
plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label = "Predicted data")
plt.legend()
plt.show()
```


![png](../assets/img/ML_DL/Linear Regression/Linear Regression from Text with Tensorflow_19_0.png)