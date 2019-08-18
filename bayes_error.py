'''
Bayes error:
	This script contains Bayes classifier to compute bayes error
'''

import math
import numpy as np

class bayes_classification:
    def __init__(self,X,y):
        self.X = X                                  # Current samples
        self.y= y                                   # Current labels
        self.n_features = len(X[0])                 # number of features
        self.n_classes = len(np.unique(y))          # number of unique classes
        self.n_samples = len(X)                     # number of samples
        self.class_attribute = self.preprocess()    # mean and std deviation seperated by class

    # Given X and y calculate and assign the class attribute
    def preprocess(self, X=None,y=None):
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        else:
            self.X = X
            self.y = y
            self.n_features = len(X[0])
            self.n_classes = len(np.unique(y))
            self.n_samples = len(X)

        dict_ = {}
        seperated_class = self.seperate_by_class()
        for class_val, features in seperated_class.items():
            dict_[class_val] = self.calculate_attribute(features)
        
        self.class_attribute = dict_
        return dict_
        
    # Given a list of samples calculate mean and standard deviation
    def calculate_attribute(self,X=None):
        if X is None:
            X = self.X
        list_ = []
        for i in range(self.n_features):
            mean = np.mean(X[:,i])
            stdev = np.std(X[:,i])
            list_.append([mean,stdev])
        return list_       
    
    # Seperate a big chunck of data to distinct classes 
    def seperate_by_class(self, X=None,y=None):
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        seperate = {}
        for i in range(self.n_classes):
            seperate[i] = X[np.where(self.y==i)[0]]
        return seperate
    
    # given mean and standard deviation calculate the gaussian probability 
    def gaussisan_probability(self,x, mean, stdev):
        exp = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
        return (1 / (math.sqrt(2*math.pi) * stdev)) * exp
    
    # given a sample calculate its probability of classifying into each class
    def calculate_class_prob(self, input_):
        class_probability = {}
        for class_val, attributes in self.class_attribute.items():
            class_probability[class_val] = 1
            for i in range(len(attributes)):
                mean, stdev = attributes[i]
                class_probability[class_val] *= self.gaussisan_probability(input_[i],mean,stdev)
        return class_probability
    
    # Predict the sample to be classified which class
    def predict(self, input_):
        class_probability = self.calculate_class_prob(input_)
        bestProb = 0
        bestClass = None
        for class_val, probability in class_probability.items():
            if bestClass is None or probability > bestProb:
                bestClass = class_val
                bestProb = probability
        return bestClass

    # Cross one out validation
    def bayes_error(self):
        error = 0
        for i in range(len(self.X)):
            original_X = self.X
            original_y = self.y
            x_trainset = np.delete(self.X, i,0)
            x_validate = self.X[i,:]
            y_trainset = np.delete(self.y,i)
            y_validate = self.y[i]

            self.preprocess(x_trainset, y_trainset)
            if self.predict(x_validate) != y_validate:
                error+=1
            self.preprocess(original_X, original_y)
    
        return error/self.n_samples