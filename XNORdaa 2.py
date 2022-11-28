# importing Python library
import numpy as np
  
# define Unit Step Function
def UnitStep(v):
    if v >= 0:
        return 1
    else:
        return 0
  
# design Perceptron Model
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


