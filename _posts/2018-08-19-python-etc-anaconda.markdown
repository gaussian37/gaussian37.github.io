---
layout: post
title: Necessary command for anaconda 
date: 2018-08-19 13:46:00
img: python/etc/necessary-command-for-anaconda/anaconda.png
categories: [python-etc] 
tags: [python, anaconda, conda] # add tag
---

### 1.Conda Basic

| Command 	| Description 	|
|:---------------------------------------:	|:-----------------------------------------------:	|
| conda info 	| Verify conda is installed, check version number 	|
| conda update conda 	| Update conda to the current version 	|
| conda install PACKAGENAME 	| Install a package included in Anaconda 	|
| conda update PACKAGENAME 	| Update any installed program 	|
| COMMANDNAME --help conda install --help 	| Command line help 	|

<br>

### 2.Using Environments

| Command 	| Description 	|
|:---------------------------------------------------------:	|:--------------------------------------------------------------:	|
| conda create --name py35 python=3.5 	| Create a new environment named py35, install Python 3.5 	|
| WINDOWS: activate py35, LINUX/macOS: source activate py35 	| Activate the new environment to use it 	|
| WINDOWS: deactivate, macOS/LINUX: source deactivate 	| Deactivate the current environment 	|
| conda env list 	| Get a list of all my environments 	|
| conda list 	| List all packages and versions installed in active environment 	|

<br>
`conda env list --name envs_name` shows package only in envs_name.

### 3.Installing and updating packages

| Command 	| Description 	|
|:-------------------------------------------:	|:----------------------------------------------------------------------------------:	|
| conda install jupyter 	| Install a new package (Jupyter Notebook) in the active environment 	|
| jupyter-notebook 	| Run an installed package (Jupyter Notebook) 	|
| conda update scikit-learn 	| Update a package in the current environment 	|
| conda install --channel conda-forge boltons 	| Install a package (boltons) from a specific channel (conda-forge) 	|
| pip install boltons 	| Install a package directly from PyPI into the current active environment using pip 	|
| conda remove --name bio-env toolz boltons 	| Remove one or more packages (toolz, boltons) from a specific environment (bio-env) 	|

<br>

### 4.Specifying version numbers

| Command 	| Description 	|
|:---------------------:	|:---------------------------------------------------:	|
| numpy==1.11 	| Exact : 1.11.0 (Never use numpy = 1.11, it's fuzzy) 	|
| "numpy>=1.11" 	| Greater than or equal to : 1.11.0 or higher 	|
| "numpy>=1.8,<2" 	| AND : 1.8, 1.9, not 2.0 	|