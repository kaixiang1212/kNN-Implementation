'''
Generator:
	This file contain generator class which is able to generate data according to the rule and desired data size
'''

import random
import numpy as np

class data_generator:
    def __init__(self, n_samples, n_features, n_informative=2 ,n_classes=2, weights=None, random_state=0):
        self.n_samples = n_samples                          # number of samples to generate
        self.n_features = n_features                        # number of features
        self.n_classes = n_classes                          # number of classes
        self.n_informative = n_informative                  # number of features that are informative
        if weights is not None:
            if len(weights) == n_classes:
                self.weights = weights                      # a list of weight of classes []
        else :
            self.weights=None
        self.random_state = random_state 
        if random_state:
            random.seed(random_state)                       # random seed
        self.rules = []                                     # a list of rules

    # -1 for <
    # 1 for >
    def add_rule(self, nth_feature, option, rate, y, probability=0.7):
        if not self.check_rule(nth_feature):
            return False
        elif (option == -1 or option == 1):
            self.rules.append((nth_feature,option,rate,y, probability))
            return True
        else:
            return False

    # return true if there exist a rule for nth feature, false otherwise
    def check_rule(self, nth_feature):
        if len(self.rules) > self.n_informative or nth_feature >= self.n_features:
            return False
        for a in self.rules:
            if a[0] == nth_feature:
                return False
        return True

    # generate random rules for data
    def generate_rules(self):
        rule_left = self.n_informative - len(self.rules)
        while rule_left != 0:
            nth_feature = random.randint(0, self.n_features-1)
            if not self.check_rule(nth_feature):
                pass
            option = random.choice([-1,1])
            rate = random.uniform(-1,1)
            y = random.randint(0,self.n_classes-1)
            if self.add_rule(nth_feature,option,rate,y):
                rule_left-=1

    # generate labels
    def _generate_y(self):
        y = []
        if self.weights != None:
            # set up number of class according to weight
            for i in range(self.n_classes):
                weight = self.weights[i] / np.sum(self.weights)
                weight = weight* self.n_samples
                for j in range(int(weight)):
                    y.append(i)
        else :
            for i in range(self.n_classes):
                weight = self.n_samples / self.n_classes
                for j in range(int(weight)):
                    y.append(i)
        while len(y) != self.n_samples:
            y.append(0)

        random.shuffle(y)
        return np.array(y)

    # generate feature based on the rule and label
    def _generate_X(self, y):
        X = []
        if self.random_state:
            np.random.seed(self.random_state)

        for i in range(len(y)):
            i_class = y[i]
            temp = np.random.uniform(0,1,self.n_features)
            # Override with Rules 
            for [nth_feature, option, rate, y_class, probability] in self.rules:
                if int(random.uniform(0,1)*100) in range(int(probability*100),100):
                    pass
                if y_class != i_class:
                    option = -option
                if option == -1:                                            # smaller than
                    temp[nth_feature] = random.uniform(0,rate)
                elif option == 1:                                           # greater than
                    temp[nth_feature] = random.uniform(rate,1)
            X.append(temp)
        return np.array(X)

    # generate a dataset
    def generate_data(self):
        y = self._generate_y()
        X = self._generate_X(y)
        return X, y

    # return a list of rules in string format
    def get_rules(self):
        rules = []
        for [nth_feature, option, rate, y_class, probability] in self.rules:
            if option == 1:
                rules.append("p( y = %d | x%d > %.2f ) = %.2f" %(y_class,nth_feature,rate,probability))
            elif option == -1:
                rules.append("p( y = %d | x%d < %.2f ) = %.2f" %(y_class,nth_feature,rate,probability))
        return rules