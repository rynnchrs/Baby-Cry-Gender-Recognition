# Baby-Cry-Gender-Recognition
Gender Recognition of babies cry using Machine Learning Algorithms(Support Vector Machine and K-Nearest Neighbors) and Signal Processing(Fast Fourier Transform and Discrete Harley Transform)


About
======
Machine Learning Algorithms used for Prediction:
- Support Vector Machine
- K-Nearest Neighbor

Signal Pre-processing used:
- Fast Fourier Transform
- Discrete Hartley Transform (only uses the real values of Fast Fourier Transform)

Length of wavefile used in training and testing:
- 13000 data (you can use PCA for lessening the data)

Installation
============

1. Use virtualenv in installing libraries and dependencies.
2. Install dependencies and libraries via pip.


To install using `pip`:

    $ pip3 install requirements.txt

  
  
Dataset:
========  
  -datasets are on the Dataset folder. 16 sample cries for boy and girl are used (More dataset, more accurate)
  
  
  
Training:
=========
1. Training consists of boy and girl dataset.
2. Either use Support Vector Machine or K-Nearest Neighbor for the Machine Learning Algorithm.
3. Either use Fast Fourier Transform or Discrete Harley Transform for the Signal Processing.



Usage:
======

**Cut the wave files**

`$ python3 cutter.py`
  
  -cuts the wave files into desired length to be used as dataset or testing sample.
  Note: wave files must be in the same length.
  
**Predict**

`$ python3 predict.py`
  
  -predict the testing sample wave file. Just input the location path of the wave file to be predicted
  
  

Sample Output:
==============

![Optional Text](../master/imagesample.png)

![Optional Text](../master/imagesample2.png)

- Frequency Sampling
- Channels
- Complete Sampling Length
- Length in seconds of the wave file
- Time response of the program
