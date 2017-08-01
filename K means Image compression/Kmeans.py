import random
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
import pandas as pd

class KMeans:
    
    def __init__(self, nb_clusters = 3):
        self.nb_clusters = nb_clusters
        self.cluster_centers = []
        
        
    def init_centers(self, x):
        #randomly initialize cluster centers from the dataset itself
        for i in range(self.nb_clusters):
            index = random.randint(0, x.shape[0]-1)
            self.cluster_centers.append(x[index])
        self.cluster_centers = np.array(self.cluster_centers).astype(float)
        self.assignments = np.zeros(x.shape[0])
    
    def dist(self, a1, a2):
        #Euclidian distance
        return (sum((a1 - a2)**2))**(1/2)
    
    def cluster_assignment(self, x):
        for i in range(x.shape[0]):
            #Find distance of the training sample i from each cluster
            dists = [0]*self.nb_clusters
            for j in range(self.nb_clusters):
                #for each cluster center ,find distance of the sample
                if type(x[i]) != np.ndarray:
                    dists[j] = self.dist(np.array([self.cluster_centers[j]]), np.array([x[i]]) )
                else:
                    dists[j] = self.dist(self.cluster_centers[j], x[i] )
            self.assignments[i] = dists.index(min(dists))
                    
    def move_centroid(self, x):
        #Find the average of each of the points in every cluster
        self.cluster_centers = np.zeros(self.cluster_centers.shape)
        class_count = np.zeros(self.nb_clusters)
        for i in range(x.shape[0]):
            self.cluster_centers[int(self.assignments[i])] = self.cluster_centers[int(self.assignments[i])]+ x[i]
            class_count[int(self.assignments[i])] += 1
        for i in range(self.nb_clusters):
            self.cluster_centers[i] = self.cluster_centers[i]/int(class_count[i])
        
    
    def fit(self, x):
        self.init_centers(x)
        for i in range(8):
            self.cluster_assignment(x)
            self.move_centroid(x)
        #self.plot(x) for 2D data, with 2 clusters
    
    def transform(self, x):
        for i in range(x.shape[0]):
            x[i] = self.cluster_centers[int(self.assignments[i])]
        return x
    
    def plot(self, x):
        plt.scatter(x[:,0], x[:,1])
        for _ in range(x.shape[0]):
            if self.assignments[_] == 0:
                plt.plot(x[_][0],x[_][1],'or')
            else:
                plt.plot(x[_][0],x[_][1],'ob')
        plt.show()
        
