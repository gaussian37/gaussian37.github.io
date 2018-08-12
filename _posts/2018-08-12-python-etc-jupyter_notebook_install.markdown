---
layout: post
title: All about jupyter notebook
date: 2018-08-12 10:30:00
img: python/etc/all-about-jupyter-notebook/jupyter.png
categories: [python-etc] 
tags: [python, jupyter] # add tag
---

+ Useful command in jupyter notebook
    - %command : run **one** line in selected cell
    - %%command : run **all** line in selected cell

|      command      |                             funtion                            |
|:-----------------:|:--------------------------------------------------------------:|
| %magic            | print detail help of all magic function                        |
| %timeit statement | print average run time after executing statement several times |
| %%time            | print run time of selected cell                                |
| %pdb              | enter into debugger if exception happens                       |
| %run script.py    | run python script.py in jupyter notebook                       |


+ Useful shortcut in jupyter notebook
In command mode
    - y : to code mode
    - m : to markdown mode
    - a : insert cell above
    - b : insert cell below
    - x : cut selected cells
    - c : copy selected cells
    - d,d : delete selected cells
    - shift + m : merge selected cells or current cell with cell below if only one cell selected.
    
In edit mode
    - ctrl + shift + - : split cell
    
    
+ jupyter notebook extension install
enter the below command consecutively
    1. pip install jupyter_contrib_nbextensions
    2. jupyter contrib nbextension install --user
    3. pip install jupyter_nbextensions_configurator
    
![nbextension](../assets/img/python/etc/all-about-jupyter-notebook/nbextensions.png)
    - Variable Inspector
    - LaTeX env
    - Highlight
    - ...
