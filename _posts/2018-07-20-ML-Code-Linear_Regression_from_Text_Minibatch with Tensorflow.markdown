---
layout: post
title: Linear_Regression_from_Text_Minibatch with Tensorflow
date: 2018-07-20 21:00:00
img: ML_DL/tensorflow.png # Add image post (optional)
categories: [machine-learning-code] 
tags: [Linear Regression, Tensorflow] # add tag
---

+ github : [Linear_Regression_from_Text_Minibatch with Tensorflow](https://nbviewer.jupyter.org/github/gaussian37/Deep-Learning/blob/master/Library/Tensorflow/Linear%20Regression/Linear_Regression_from_Text_Minibatch%20with%20Tensorflow.ipynb)
+ data : [data](https://github.com/gaussian37/Deep-Learning/blob/master/Library/Tensorflow/Linear%20Regression/data/birth_life_2010.txt)

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
```

Read the data file
Linear_Regression_from_Text_Minibatch with Tensorflow.ipynb <br>
data <br>
└──── birth_life_2010.txt <br>

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
batch_size = 10
n_epoch = 1000
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

### Step 1 : read in data from the .txt file


```python
data, n_samples = read_birth_file_data(DATA_FILE)
```

### Step 2 : create placeholders for X (birth rate) and Y (life expectancy)


```python
X = tf.placeholder(tf.float32, [None,], name = "X")
Y = tf.placeholder(tf.float32, [None,], name = "Y")
```

### Step 3 : create weight and bias, initialized to 0


```python
w = tf.get_variable("weights", initializer = tf.constant(0.0))
b = tf.get_variable("bias", initializer= tf.constant(0.0))
```

### Step 4 : build model to predict Y


```python
hypothesis = w * X + b
```

### Step 5 : use the squared error as the loss function


```python
loss = tf.reduce_mean(tf.square(hypothesis - Y), name = "loss")
```

### Step 6 : using gradient descent with learning rate of 0.001 to minimize loss


```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
```

### Step 7, 8, 9 : 


```python
writer = tf.summary.FileWriter("./graphs/linear_reg_minibatch", tf.get_default_graph())
with tf.Session() as sess:
    # Step 7 : initialize the necessary variables, in this case, w and b 
    sess.run(tf.global_variables_initializer())
    
    total_batch = int(n_samples / batch_size)
    
    # Step 8 : train the model for 100 epoch
    for epoch in range(n_epoch):
        total_loss = 0
        for i in range(total_batch):
            x_data = data[i*batch_size: (i+1)*batch_size, 0]
            y_data = data[i*batch_size: (i+1)*batch_size, 1]
            
            # Session execute optimizer and fetch values of loss
            _, _loss = sess.run([optimizer, loss], feed_dict = {X:x_data, Y:y_data})
            total_loss += _loss
        if epoch % 10 == 0:
            print("Epoch {0} : {1}".format(epoch, total_loss / total_batch))
        
    #close the writer when you're done using it
    writer.close()
    
    # Step 9 : output the values of w and b
    w_out, b_out = sess.run([w, b])
```

    Epoch 0 : 3873.261664139597
    Epoch 10 : 1177.9284186112254
    Epoch 20 : 1021.8864087556537
    Epoch 30 : 890.0951409590872
    Epoch 40 : 775.8602037931744
    Epoch 50 : 676.8043333354749
    Epoch 60 : 590.9107232344778
    Epoch 70 : 516.4300199809827
    Epoch 80 : 451.84580913342927
    Epoch 90 : 395.84292763157896
    Epoch 100 : 347.2815310829564
    Epoch 110 : 305.17327680085833
    Epoch 120 : 268.660385533383
    Epoch 130 : 236.99898047196237
    Epoch 140 : 209.54497929623253
    Epoch 150 : 185.73920280054995
    Epoch 160 : 165.096669648823
    Epoch 170 : 147.19749892385383
    Epoch 180 : 131.67683390567177
    Epoch 190 : 118.2189413371839
    Epoch 200 : 106.54922525506271
    Epoch 210 : 96.43009768034283
    Epoch 220 : 87.65597775107936
    Epoch 230 : 80.04791420384457
    Epoch 240 : 73.4511698672646
    Epoch 250 : 67.73079420390881
    Epoch 260 : 62.77094068025288
    Epoch 270 : 58.47038409584447
    Epoch 280 : 54.74108866641396
    Epoch 290 : 51.50764425177323
    Epoch 300 : 48.704090319181745
    Epoch 310 : 46.273195567883946
    Epoch 320 : 44.165450748644375
    Epoch 330 : 42.33779555872867
    Epoch 340 : 40.75325062400416
    Epoch 350 : 39.37916042930201
    Epoch 360 : 38.187887392546
    Epoch 370 : 37.155074069374486
    Epoch 380 : 36.25946812880667
    Epoch 390 : 35.48317924298738
    Epoch 400 : 34.809891951711556
    Epoch 410 : 34.22618007659912
    Epoch 420 : 33.72020962363795
    Epoch 430 : 33.28137538307592
    Epoch 440 : 32.901002381977285
    Epoch 450 : 32.57119269120066
    Epoch 460 : 32.2852144241333
    Epoch 470 : 32.037311152407995
    Epoch 480 : 31.822475784703304
    Epoch 490 : 31.636232376098633
    Epoch 500 : 31.474788565384713
    Epoch 510 : 31.334768696835166
    Epoch 520 : 31.213394114845677
    Epoch 530 : 31.10822065253007
    Epoch 540 : 31.017030565362226
    Epoch 550 : 30.938024520874023
    Epoch 560 : 30.869479480542633
    Epoch 570 : 30.81015888013338
    Epoch 580 : 30.75869665647808
    Epoch 590 : 30.714096872430098
    Epoch 600 : 30.675428139536006
    Epoch 610 : 30.641936603345368
    Epoch 620 : 30.612956247831647
    Epoch 630 : 30.587803840637207
    Epoch 640 : 30.566012081347015
    Epoch 650 : 30.54714830298173
    Epoch 660 : 30.53080463409424
    Epoch 670 : 30.516621188113565
    Epoch 680 : 30.50435929549368
    Epoch 690 : 30.493725776672363
    Epoch 700 : 30.48450419777318
    Epoch 710 : 30.476536750793457
    Epoch 720 : 30.46962652708355
    Epoch 730 : 30.46364181920102
    Epoch 740 : 30.45845357995284
    Epoch 750 : 30.453973870528372
    Epoch 760 : 30.45011625791851
    Epoch 770 : 30.446753451698704
    Epoch 780 : 30.443865324321546
    Epoch 790 : 30.441354450426605
    Epoch 800 : 30.4391652659366
    Epoch 810 : 30.437284469604492
    Epoch 820 : 30.43565815373471
    Epoch 830 : 30.43424099370053
    Epoch 840 : 30.433045939395303
    Epoch 850 : 30.432000160217285
    Epoch 860 : 30.43107630077161
    Epoch 870 : 30.43029745001542
    Epoch 880 : 30.429620692604466
    Epoch 890 : 30.42902404383609
    Epoch 900 : 30.428540731731214
    Epoch 910 : 30.42808889087878
    Epoch 920 : 30.427750135722913
    Epoch 930 : 30.427408519544098
    Epoch 940 : 30.427137123911006
    Epoch 950 : 30.42688364731638
    Epoch 960 : 30.426684680737946
    Epoch 970 : 30.42652130126953
    Epoch 980 : 30.426354056910466
    Epoch 990 : 30.426233040659053
    


```python
plt.plot(data[:,0], data[:,1], 'bo', label = 'Real data')
plt.plot(data[:,0], data[:,0]* w_out + b_out, 'r', label = "Predicted data")
plt.legend()
plt.show()
```

![png](Linear_Regression_from_Text_Minibatch%20with%20Tensorflow_files/Linear_Regression_from_Text_Minibatch%20with%20Tensorflow_20_0.png)




![png](../assets/img/ML_DL/Linear Regression/Linear Regression from Text with Tensorflow_19_0.png)