---
layout: post
title: Decision Tree (Reference ML KAIST)
date: 2017-01-02 00:00:00
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
    - Instance `X` <br>
        - Features : <Sunny, Warm, Normal, Strong, Warm, Same>
        - Label : Yes
        
    - Training dataset `D` <br>
        - A collection of observations on the instance (Collection of Xs)
        
    - Hypothesis `H` <br>
        - Potentially possible function to turn X into Y
        - h_i : <Sunny, Warm, ?, ?, ?, Same> → Yes
    
    - Target function `C` <br>
        - Unknown target function between the features and the label
        - Our objective is expressing hypothesis `H` to target function `C`.
        
        
### Graphical Representation of Function Approximation

![1](../assets/img/ml/concept/about-decision-tree/graphic.png)

- x1 : <Sunny, Warm, Normal, Strong, Warm, Same>, h1 : <Sunny, ?, ?, ?, Warm, ?> <br>
- x2 : <Sunny, Warm, Normal, Light, Warm, Same>, h2 : <Sunny, ?, ?, ?, Warm, Same> <br>
- x3 : <Sunny, Warm, Normal, Strong, Warm, Change>, h3 : <Sunny, ?, ?, Strong, Warm, ?>

+ What would be the better function approximation?
    - Generalization Vs. Specialization
    - Instance x1 & Hypothesis h1 is `Generalization`
    - Instance x3 & Hypothesis h3 is `Specialization`
    
    
### Find-S Algorithm

![Find-S](../assets/img/ml/concept/about-decision-tree/Find-S.png)

+ Find-S Algorithm
    + Initialize h to the most specific in H
    + For `instance` x in D
        + if x is positive
            + For `feature` f in O
                + if $$ f_{i} $$ in h == $$ f_{i} $$ in x
                    + Do nothing
                + else
                    + ($$ f_{i} $$ in h) ← ($$ f_{i} $$ in h) $$ \cup $$ ($$ f_{i} $$ in x)
                    
    + Return h
    
<br>

For example, <br>

+ Instances
    + $$ x_{1} $$ : <Sunny, Warm, Normal, Strong, Warm, Same>
    + $$ x_{2} $$ : <Sunny, Warm, Normal, Light, Warm, Same>
    + $$ x_{4} $$ : <Sunny, Warm, Normal, Strong, Warm, Change>
    
+ Hypothesis
    + $$ h_{0} $$ = $$ <\varnothing, \varnothing, \varnothing, \varnothing, \varnothing \varnothing> $$
    + $$ h_{1} $$ = <Sunny, Warm, Normal, Strong, Warm, Same>
    + $$ h_{1,2} $$ = <Sunny, Warm, Normal, **Don't care**, Warm, Same>
    + $$ h_{1,2,4} $$ = <Sunny, Warm, Normal, **Don't care**, Warm, **Don't care**>
    
+ Any possible problems?
    + Many possible hs, and can't determine the coverage.
    

### Version Space

+ Many hypotheses possible, and No way to find the convergence.
+ Need to setup the perimeter of the possible hypothesis.
+ The set of the possible hypotheses == Version Space (**VS**)
    - General Boundary : `G`
        - Is the set of the `maximally general` hypotheses of the version space.
    - Specific Boundary : `S`
        - Is the set of the `maximally specific` hypotheses of the version space.
    - Every hypothesis, **h**, satisfies
        - VS = {h $$ \in $$ H | $$ \exists $$ s  $$ \in $$ S, $$ \exists $$ g  $$ \in $$ G, g $$ \ge $$ h $$ \ge $$ s} <br>
          where x $$ \ge $$ y means x is more general or equal to y
  
![version_space](../assets/img/ml/concept/about-decision-tree/version_space.png)

<br>

`Version Space` is not too general and also not too specific.

## Candidate Elimination Algorithm

+ Candidate Elimination Algorithm
    - Initialize **S** to maximally specific h in H
    - Initialize **G** to maximally general h in H
    - For instance x in D
        - If y of x is positive
            - Generalize S as much as needed to **cover** o in x.
            - Remove any h in G, for which $$ h(o) \neq y $$
        - If y of x is negative
            - Specialize G as much as needed to **exclude** o in x.
            - Remove any h in S, for which $$ h(o) \eq y $$
            
+ Generate h that satisfies $$ \exists s \in S, \exists g \in G, g \ge h \ge s $$.

![candidata_elimination](../assets/img/ml/concept/about-decision-tree/candidate_elimination.png)

There are many `hs` in the H and we can guess that some of them are `True`.
In this case, we set the rule for searching the correct hypothesis. This is the `rule-based learning`.

## How to classify the next instances ?

+ Somehow, we come up with the version space.
    - A subset of **H** that satisfies the training data **D**.
+ Imagine a new instance kicks in
    - <Sunny, Warm, Normal, Strong, Cool, Change>
    - <Rainy, Cold, Normal, Light, Warm, Same>
    - <Sunny, Warm, Normal, Light, Warm, Same>
    - ...
    - Some cases are **not classified**...
    
+ How to classify these?
    - which **h** to apply from the subset?
    - Or, a classification by all of **h**s in the subset
    - How many are **h**s satisfied?
    - Sometimes, `rule-based learning` performance is terrible,,, so it's not easy to apply the hypothesis.
    
## Is this working?

+ Will the candidate-elimination algorithm converge to the correct hypothesis?
    - Converge ? → Able to select a hypothesis
    - Correct ? → The hypothesis is true in the observed system
    
+ Given the following assumption, yes and yes !
    - No observation errors, No inconsistent observations
        - Training data is error-free, noise-free.
    - No stochastic elements in the system we observe
        - Target function is deterministic
    - Full information in the observations to regenerate the system
        - Target function is contained in hypotheses set

+ However, we don't live in the perfect world!
    - Noise data : correct **h** can be removed by the noise.
    - Wrong decision factor
    



 
            
           
     
  
         
        
        
