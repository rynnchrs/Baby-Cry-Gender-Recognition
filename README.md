# Baby-Cry-Gender-Recognition
Gender Recognition of babies cry using Machine Learning Algorithms(Support Vector Machine and K-Nearest Neighbors) and Signal Processing(Fast Fourier Transform and Discrete Harley Transform)


Installation:
1. Use virtualenv in installing libraries and dependencies.
2. Install necessary packages "pip3 install -r requirements.txt".



Codes:

cutter.py
  
  -cuts the wave files into desired length to be used as dataset or testing sample.
  Note: wave files must be in the same length.
  
predict.py
  
  -predict the testing sample wave file. Just input the location path of the wave file to be predicted
  
  
Dataset:
  
  -datasets are on the Dataset folder. 16 sample cries for boy and girl are used (More dataset, more accurate)
  
  
  
Training:
1. Training consists of boy and girl dataset.
2. Either use Support Vector Machine or K-Nearest Neighbor for the Machine Learning Algorithm.
3. Either use Fast Fourier Transform or Discrete Harley Transform for the Signal Processing.

Sample Output:
![Optional Text](../master/imagesample.png)

-Frequency Sampling
-Channels
-Complete Sampling Length
-Length in seconds of the wave file
-Time response of the program
