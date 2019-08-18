'''
Error functions:
	- percentage error for numeric prediction
	- error estimation for classification
'''

import numpy as np

# Single step percentage error for numeric prediction
def error_percentage_step(pred,true):
    diff = abs(pred - true)
    frac = diff/true
    return frac
	
# Overall error calculation for classification
def error_discrete(y_preds, y_trues):
	sum = 0
	for i in range(len(y_preds)):
		if not y_preds[i] == y_trues[i]:
			sum+=1
	return sum/len(y_preds)