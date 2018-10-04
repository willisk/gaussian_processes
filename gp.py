import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, plot

class GP():

    def __init__(self):
        s0 = 1
        s1 = 1
        self.kernel = lambda x_1,x_2: s0*np.exp(-.5*s1*(x_1-x_2)**2)
        self.kernel_matrix = lambda X, Y: np.array( [[self.kernel(xi,yi) \
                for xi in X] for yi in Y] )
        self.bi = 0

    def fit(self, X, t):
        self.X = X;
        self.t = t;
        K = self.kernel_matrix(X,X)
        self.Ci = np.linalg.inv( K + np.eye(len(X))*self.bi )

    def predict(self, Y):
        k = self.kernel_matrix(self.X,Y)
        m = k.dot(self.Ci.dot(self.t))
        s = self.kernel_matrix(Y,Y) - k.dot(self.Ci.dot(k.T))
        return m, s


#noise = .1
#samples = 20

#x = np.random.uniform(-.5,6,[samples])
#t = np.sin(x) + np.array(np.random.normal(0,noise,[samples]))

gp = GP()

#fy = np.sin(fx)
#plt.plot(fx,fy,"g", lw=1)

data =  np.array([
            [-2,-.5,.8,2.3,4,5.5],
            [3,-2,8,2,1,5]
        ])

plt.plot(data[0], data[1],"bo",lw=.5)

grid_x = np.linspace(min(data[0]),max(data[0]),num=500)

gp.fit(data[0],data[1])

#vp = np.vectorize( gp.predict )
post_m, post_s = gp.predict(grid_x)

#plt.plot(grid_x,ft,"r",lw=1)
#plt.fill_between(grid_x,ft+s,ft-s,alpha=.5)
#plt.plot(grid_x,st.multivariate_normal.rvs(ft,s),lw = 1)

#plt.plot(grid_x,ft+s,"r--",lw = 1)
#plt.plot(grid_x,ft-s,"r--",lw = 1)

#traces = go.Scatter( x=grid_x, y=st.multivariate_normal.rvs(ft,s) )

post = np.random.multivariate_normal(post_m, post_s, size=1000)

for y in post:
    plt.plot(grid_x, y,c='k', alpha=.01)

plt.show()



#marker = dict( size=12, color='rgb(255,0,0)' )
#trace1 = go.Scatter( x=data[0], y=data[1], 
#        mode='markers', marker=marker )
#
#traces = go.Scatter( x=grid_x, y=st.multivariate_normal.rvs(ft,s) )
#
#plot([trace1,traces])
