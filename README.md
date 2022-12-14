# Design-and-Analysis-of-Algorithm
## Assignment 2

Name : Rajesh Thakare

Sec : 5-A

Roll : 54

## Problem Statement :  Given an application and its implementation to demonstrate the implementation of XNOR gate [Digital Ckts]

## Logic/Thoery: 
Implementation of Perceptron Algorithm for XNOR logic gate with 2-bit binary input.Perceptron is Machine Learning algorithm for supervised learning of various binary classification tasks.Binary classifiers are defined as the function that helps in deciding whether input data can be represented as vectors of numbers and belongs to some specific class.Perceptron is considered as a single-layer neural network that consists of four main parameters named input values (Input nodes), weights and Bias, net sum, and an activation function. The perceptron model begins with the multiplication of all input values and their weights, then adds these values together to create the weighted sum. Then this weighted sum is applied to the activation function 'f' to obtain the desired output. This activation function is also known as the step and function is represented by 'f'.

![image](https://user-images.githubusercontent.com/108029540/204028908-9c325cc6-3f3b-44d5-a552-50b99df2910d.png)










The Perceptron Model implements the following function:

![Screenshot (355)](https://user-images.githubusercontent.com/108029540/204022012-7c45752d-6841-44ac-a000-5268c4bdeefd.png)








## Approach :     


![truth_xnor](https://user-images.githubusercontent.com/108029540/204030724-87715028-5fff-400e-bcb0-5cec00d6fdbf.png)


We can observe that,XNOR(x1,x2)=OR(NOT(OR(x1,x2)),AND(x1,x2))

Step 1: Now for the corresponding weight vector w:(w1,w2) of the input vector x:(x1,x2) to the OR and AND node, the associated Perceptron Function can be defined as:
           y1=(w1x1+w2x2+bOR)
           y2=(w1x1+w2x2+bAND)
           
Step 2:The output (y1) from the OR node will be inputed to the NOT node with weight w(NOT) and the associated Perceptron Function can be defined as:
          y3=(w(NOT)y1+bNOT)
          
Step 3:The output (y2) from the AND node and the output(y3) from NOT node as mentioned in Step2 will be inputed to the OR node with weight (w OR1, w OR2).Then the corresponding output $\boldsymbol{\hat{y}}$ is the final output of the XNOR logic function. The associated Perceptron Function can be defined as:
                            y=(w OR1 y3+ w OR2 y2 +b OR)
                            

![XN_p](https://user-images.githubusercontent.com/108029540/204022075-b14d73b8-4520-4b84-a474-c1b0e7869583.png)
                            
                            
     
 For the implementation, the weight parameters are considered to be w1=1,w2=1,w NOT=-1,w OR2=1 and the bias parameters are b AND=-1.5, b OR=-0.5 , b NOT=0.5
 

```
# importing Python library
import numpy as np
  
# dUnit Step Function
def UnitStep(v):
    if v >= 0:
        return 1
    else:
        return 0
  
#  Perceptron Model Function
def PerceptronModel(x, w, b):
    v = np.dot(w, x) + b
    y = UnitStep(v)
    return y
  
# NOT Logic Function
# wNOT = -1, bNOT = 0.5
def NOT_Function(x):
    wNOT = -1
    bNOT = 0.5
    return PerceptronModel(x, wNOT, bNOT)
  
# AND Logic Function
# w1 = 1, w2 = 1, bAND = -1.5
def AND_Function(x):
    w = np.array([1, 1])
    bAND = -1.5
    return PerceptronModel(x, w, bAND)
  
# OR Logic Function
# here w1 = wOR1 = 1, 
# w2 = wOR2 = 1, bOR = -0.5
def OR_Function(x):
    w = np.array([1, 1])
    bOR = -0.5
    return PerceptronModel(x, w, bOR)
  
# XNOR Logic Function
# with AND, OR and NOT  
# function calls in sequence
def XNOR_Function(x):
    y1 = OR_Function(x)
    y2 = AND_Function(x)
    y3 = NOT_Function(y1)
    finalvalue = np.array([y2, y3])
    Output = OR_Function(finalvalue)
    return Output
  
# testing the Perceptron Model
test1 = np.array([0, 1])
test2 = np.array([1, 1])
test3 = np.array([0, 0])
test4 = np.array([1, 0])
  
print("XNOR({}, {}) = {}".format(0, 1, XNOR_Function(test1)))
print("XNOR({}, {}) = {}".format(1, 1, XNOR_Function(test2)))
print("XNOR({}, {}) = {}".format(0, 0, XNOR_Function(test3)))
print("XNOR({}, {}) = {}".format(1, 0, XNOR_Function(test4)))
```

Output :

XNOR(0, 1) = 0

XNOR(1, 1) = 1

XNOR(0, 0) = 1

XNOR(1, 0) = 0


The model predicted output  for each of the test inputs are exactly matched with the XNOR logic gate conventional output  according to the truth table.
Hence, it is verified that the perceptron algorithm for XNOR logic gate is correctly implemented.


                            
                            

                            
        


















