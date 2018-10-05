import numpy as np
from matplotlib import pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go

class GP():

    def __init__(self):
        self.bi = 0
        self.kernel = self.kernel_shape(1,1,2)
        self.kernel_matrix = lambda X, Y: np.array( [[self.kernel(xi,yi) \
                for xi in X] for yi in Y] )

    def kernel_shape(self,s0, s1, s2):
        return lambda x1, x2: s0 * np.exp(-.5*s1*np.abs(x1-x2)**s2)

    def fit(self, X, t):
        self.X = X;
        self.t = t;
        K = self.kernel_matrix(X,X)
        self.Ci = np.linalg.inv( K + np.eye(len(K)) * self.bi )

    def predict(self, Y):
        self.Y = Y
        k = self.kernel_matrix(self.X,Y)
        m = k.dot(self.Ci.dot(self.t))
        s = self.kernel_matrix(Y,Y) - k.dot(self.Ci.dot(k.T))
        return m, s
    
    def update(self, bi, s0, s1, s2):
        self.bi = bi
        self.kernel = self.kernel_shape(s0,s1,s2)
        self.fit(self.X,self.t)
        return self.predict(self.Y)

