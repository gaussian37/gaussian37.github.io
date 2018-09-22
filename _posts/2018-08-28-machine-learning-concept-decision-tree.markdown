---
layout: post
title: About Decision Tree  
date: 2018-08-28 22:10:00
img: ml/concept/about-decision-tree/decision-tree.png
categories: [ml-concept] 
tags: [machine learning, decision tree] # add tag
---

Before dividing into Decision Tree, what is machine learning? <br>
+ A computer program is said that
    - learn from experience E
    - With respect to some class of tasks T
    - And performance measure P, if its performance at tasks in T, as measured by P, improves with experience E
    
  

### A Perfect World for Rule Based Learning   

Imagine a perfect world with
+ No observation errors, No inconsistent observations.
    - (Training data is error-free, noise-free)
+ No stochastic elements in the system we observe.
    - (Target function is deterministic)
+ Full information in the observations to regenerate the system
    - (Target function is contained in hypotheses set)
    
Observation of the people
+ Sky, Temp, Humid, Wind, Water, Forecast → EnjoySport

|  Sky  | Temp |  Humid |  Wind  | Water | Forecast | EnjoySport |
|:-----:|:----:|:------:|:------:|:-----:|:-------:|:----------:|
| Sunny | Warm | Normal | Strong |  Warm |   Same  |     Yes    |
| Sunny | Warm |  High  | Strong |  Warm |   Same  |     Yes    |
| Rainy | Cold |  High  | Strong |  Warm |  Change |     No     |
| Sunny | Warm |  High  | Strong |  Cool |  Change |     Yes    |

<br>

### Function Approximation

Machine learning is relative to applying function approximation well.

+ Machine Learning ?
    - The effort of producing a better approximate function

+ In the perfect world of EnjoySport
    - Instance **X** <br>
        - Features : <Sunny, Warm, Normal, Strong, Warm, Same>
        - Label : <Yes>
        
    - Training dataset **D** <br>
        - A collection of observations o n the instance
        
    - Hypothesis **H** <br>
        - Potentially possible function to turn X into Y
        - h_i : <Sunny, Warm, ?, ?, ?, Same> → Yes
    
    - Target function **C** <br>
        - Unknown target function between the features and the label
        
        
### Graphical Representation of Function Approximation

![1](../assets/img/ml/concept/about-decision-tree/graphic.png)

- x1 : <Sunny, Warm, Normal, Strong, Warm, Same>, h1 : <Sunny, ?, ?, ?, Warm, ?> <br>
- x2 : <Sunny, Warm, Normal, Light, Warm, Same>, h2 : <Sunny, ?, ?, ?, Warm, Same> <br>
- x3 : <Sunny, Warm, Normal, Strong, Warm, Change>, h3 : <Sunny, ?, ?, Strong, Warm, ?>

+ What would be the better function approximation?
    - Generalization Vs. Specialization
    
    



