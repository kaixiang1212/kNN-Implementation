import numpy as np
import pandas as pd
import math
# import matplotlib.pyplot as plt

# knn classifier
class knnClassifier:
	def __init__(self, train, labels, k , dis_mode = 'e', pred_mode = 'vote'):
		self.trainSet = train			  # train set
		self.labels = np.array(labels)	 # labels
		self.featureSize = len(train[0])   # number of features in the set
		self.disMode = dis_mode			# euclidean distance or manhattan distance
		self.predMode = pred_mode		  # vote (for classification) or avg (for real value prediction)
		self.setK(k)					   # set k value
		
	def setK(self, k):
		self.k = k
		
	# compute euclidean distance
	def euclideanDistance(self, instance1, instance2):
		distance = 0
		for i in range(self.featureSize):
			distance += pow((instance1[i]-instance2[i]),2)
		return math.sqrt(distance)

	# compute manhattan distance
	def manhattanDistance(self, instance1, instance2):
		distance = 0
		for i in range(self.featureSize):
			distance += abs(instance1[i]-instance2[i])
		return distance

	# calculate the distance
	def calDistance(self, instance1, instance2):
		if self.disMode == 'e':
			dis = self.euclideanDistance(instance1, instance2)
		elif self.disMode == 'm':
			dis = self.manhattanDistance(instance1, instance2)
		else:
			print("Invalid mode, please enter 'e' or 'm'")
			return
		return dis
	
	# Get the neighbours of an instance from the train set
	# return the indexs of the nearest neighbours
	def getNeighbour(self,instance):
		distance = []
		train_len = len(self.trainSet)
		
		# for each element in train set, cal the distance
		# print(self.trainSet)
		for i in range(train_len):
			curDis = self.calDistance(self.trainSet[i],instance)
			distance.append([i,curDis])
			
		# sort the distance
		distance = sorted(distance, key=lambda x:x[1])
		
		# get the first k result
		result = np.array(distance)
		result = [int(x[0]) for x in result]
		result = result[0:self.k]
		return result
	
	# predict using the avg method
	def predict_avg(self,instance):
		
		top_dis = self.getNeighbour(instance)
		target_list = self.labels[top_dis]
		return np.mean(target_list)
	   
	# predict using the voting method
	def predict_vote(self,instance):
		top_dis = self.getNeighbour(instance)
		# print("----- in pred vote -----")
		# print(instance)
		# print(top_dis)
		# print(self.labels)
		target_list = self.labels[top_dis]
		# print(target_list)
		unique_list = np.unique(target_list,return_counts = True)
		# print(unique_list)
		max_ind = np.argmax(unique_list[1])
		pred = unique_list[0][max_ind]
		# print("pred is "+str(pred))
		
		return pred
		
	# predict
	def predict(self,x_test):
		test_len = len(x_test)
		res_list = []
		
		for i in range(test_len):
			if self.predMode == 'vote':
				res = self.predict_vote(x_test[i])
			elif self.predMode == 'avg':
				res = self.predict_avg(x_test[i])
			res_list.append(res)
		return res_list
		
		
# ============== wnn classifier ==================
# similar to knn but "predict_avg" and "predict_vote" are different

class wnnClassifier:
	def __init__(self, train, labels, k , dis_mode = 'e', pred_mode = 'vote',dis_threshold = 0.000001):
		self.trainSet = train
		self.labels = np.array(labels)
		self.featureSize = len(train[0])
		self.disMode = dis_mode
		self.predMode = pred_mode
		self.disThreshold = dis_threshold
		self.setK(k)
		
	def setK(self, k):
		self.k = k
		
	def euclideanDistance(self, instance1, instance2):
		distance = 0
		for i in range(self.featureSize):
			distance += pow((instance1[i]-instance2[i]),2)
		return math.sqrt(distance)

	def manhattanDistance(self, instance1, instance2):
		distance = 0
		for i in range(self.featureSize):
			distance += abs(instance1[i]-instance2[i])
		return distance

	def calDistance(self, instance1, instance2):
		if self.disMode == 'e':
			dis = self.euclideanDistance(instance1, instance2)
		elif self.disMode == 'm':
			dis = self.manhattanDistance(instance1, instance2)
		else:
			print("Invalid mode, please enter 'e' or 'm'")
			return
		return dis
	
	# Get the neighbours of an instance from the train set
	# return the index and weight of the nearest neighbours (DIFF FROM KNN)
	def getNeighbour(self,instance):
		distance = []
		train_len = len(self.trainSet)
		
		for i in range(train_len):
			curDis = self.calDistance(self.trainSet[i],instance)
			# Prevent existance of 1/0
			if curDis < self.disThreshold:
				weight = 1/self.disThreshold
			else:
				weight = 1/curDis
			
			distance.append([i,curDis, weight])

		distance = sorted(distance, key=lambda x:x[1])
		result = np.array(distance)
		result = [[int(x[0]),x[2]] for x in result]
		result = result[0:self.k]
		return result
	
	# return: prediction = sum(weight*label) / sum(weight)
	def predict_avg(self,instance):
		top_dis = self.getNeighbour(instance)
		top_ind = [x[0] for x in top_dis]
		target_list = self.labels[top_ind]
		prod_list = []
		
		# weight sum = 0
		sum_weight = 0
		for i in range(len(top_ind)):
			
			# calculate weight * current label
			curProd = top_dis[i][1]*target_list[i]
			
			# add to weight sum
			sum_weight = sum_weight + top_dis[i][1]
			
			# add to list
			prod_list.append(curProd)
		
		res = np.sum(prod_list)/sum_weight
		return res
		
	def predict_vote(self,instance):
		top_dis = self.getNeighbour(instance)
		top_ind = [x[0] for x in top_dis]
		top_weights = [x[1] for x in top_dis]
		target_list = self.labels[top_ind]
		
		# build a dictionary containing: {labels: weigted sum}
		target_dict = {}
		for i in range(len(target_list)):
			if target_list[i] in target_dict:
				target_dict[target_list[i]] += top_weights[i]
			else:
				target_dict[target_list[i]] = top_weights[i]
			# print(target_dict)
		res = max(target_dict, key=target_dict.get)
		return res

	def predict(self,x_test):
		test_len = len(x_test)
		res_list = []
		
		for i in range(test_len):
			if self.predMode == 'vote':
				res = self.predict_vote(x_test[i])
			elif self.predMode == 'avg':
				res = self.predict_avg(x_test[i])
				
			#print(res)
			res_list.append(res)
		return res_list