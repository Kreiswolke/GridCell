# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 19:28:58 2015

@author: Oliver
"""


import matplotlib # plotting library
import matplotlib.mlab as mlab # matlab compatibility functions
from matplotlib.backends import backend_agg as agg # raster backend
import pandas # data analysis library
import numpy # numerical routines





fig = matplotlib.figure.Figure() # create the figure
agg.FigureCanvasAgg(fig) # attach the rasterizer
ax = fig.add_subplot(1, 1, 1) # make axes to plot on
cmap = matplotlib.cm.get_cmap("jet")
pts = ax.scatter(x_pref[:,0], x_pref[:,0], s=60, c=Weights[:,5], cmap=cmap,
linewidth=0)
cbar = fig.colorbar(pts, ax=ax)
fig.axes[-1].set_ylabel("Z")
ax.grid()
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xlim(0,10)
ax.set_ylim(0,10)
ax.set_title("Scatter Plot of Non-Gridded 2D Data")
show()