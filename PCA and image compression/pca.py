import numpy as np
from numpy.linalg import svd #for singular value decomposition

class PCA:
    
    def __init__(self, nb_components):
        self.nb_components = nb_components
        
    def normalization(self, x):
        #For each column/feature, find average and subtract from each of the sample
        uj = np.average(x, axis = 0)
        x = x - uj
        return x
    
    def singular_value_decomposition(self, x):
        #First find covariance matrix
        self.n = x.shape[1]
        self.m = x.shape[0]
        cov = np.matmul(np.transpose(x), x)  #X'X
        cov = cov/self.m
        #Now perform svd on covariance matrix
        self.u, s, v = svd(cov)
        self.S = s
        #Now extract the required number of vectors from u
        self.u = self.u[:, 0 : self.nb_components]
        #Columns of u are size (nb_components,1) vectors on which data will be projected
        self.Z = np.matmul(x, self.u)
        return (self.Z, v)
    
    def reconstruct(self):
        return np.matmul(self.Z, np.transpose(self.u))
    
    def get_variance_score(self):
        return np.sum(self.S[0: self.nb_components]) / np.sum(self.S[0: self.n])
    
    def fit_transform(self, x):
        #normalize
        x= self.normalization(x)
        
        #perform SVD
        Z, v = self.singular_value_decomposition(x)
        return Z