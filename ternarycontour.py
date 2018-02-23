from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def V(x,y,z):
     return np.cos(10*x) + np.cos(10*y) + np.cos(10*z) + 2*(x**2 + y**2 + z**2)

X,Y = np.mgrid[-1:1:100j, -1:1:100j]
Z_vals = [ -0.5, 0, 0.9 ]
num_subplots = len( Z_vals)

fig = plt.figure( figsize=(10,4 ) )
for i,z in enumerate( Z_vals) :
    ax = fig.add_subplot(1 , num_subplots , i+1, projection='3d', axisbg='gray')
    ax.contour(X, Y, V(X,Y,z) ,cmap=cm.gnuplot)
    ax.set_title('z = %.2f'%z,fontsize=30)
fig.savefig('contours.png', facecolor='grey', edgecolor='none')