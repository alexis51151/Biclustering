# -*- coding: utf-8 -*-
"""
@author: Alexis Le Glaunec
Constrained K-means algorithm
(idea from Data Clustering with Cluster Size Constraints Using 
a Modified k-means Algorithm paper)
"""
import numpy as np # For matrix calculation
import random # For initialization


class KmeansConstrained:
    def __init__(self, X, Nmax, N, K, i):
        self.X = X # Matrix of objects
        self.Nmax = Nmax # Zeta in article ; max nb of object for a cluster
        self.N = N # Number of objects (can be deduced from X though)
        self.K = K # Number of clusters
        self.i = i # nb of iterations
        self.centroids = [[0,0] for k in range(K)]  # (centroid point, number of elements) for each cluster
        self.map = [-1 for i in range(N)] # Mapping between an object (index) and its cluster (value)
        
    def initialization(self):
        if self.N == 0:
            return 1 # Error
        rand_list = [i for i in range(self.N)] # random list with, as value, the index of each object
        for k in range(self.K): # Initializing each cluster with an object being the centroid
            pick = random.randint(len(rand_list))
            self.map[rand_list[pick]] = -k # the rand_list[pick]-th object goes to cluster k
            self.centroids[k] = (self.X[rand_list[pick]], 1) # And this point defines cluster k centroid
            rand_list.pop(pick) # We cannot choose it for another cluster
        return 0 # Success
    def assignment(self, n):
        for i in range(self.N): # For each object
            values = []
            for k in range(self.K):
                values.append([np.linalg.norm(np.diff(self.X[i],self.centroids[k][0])), k])
            values.sort(key= lambda e : e[0])
            for element in values: # We try to add the object into each cluster (from the nearest)
                size = self.centroids[element[1]][1]
                if(size < self.Nmax): # Update centroid and size of the cluster
                    self.centroids[element[1]][0] = 1/(size+1)*(np.add(size*self.centroids[element[1]][0], self.X[i]))
                    self.centroids[element[1]][1] += 1
                    break
        if(n == self.i):
            return (self.X, self.map)
        else:
            for element in self.map: # Map reinitialization
                if element >= 0:
                    element = -1
            return self.assignment(self, n+1)
        
        
