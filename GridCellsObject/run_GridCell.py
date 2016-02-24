# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 11:30:17 2015

@author: Oliver
"""
# This is the script to run the grid cell learning
import Grid_cell
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import cProfile


# Setting Initial parameters
t_0=0.0
t_max=2
dt=0.01
sidelength=1.25
N_in=225
N_out=100

Unit=10
sig_p= 0.05
num_pix_side=128
delta_r= sidelength/(1.0*num_pix_side)

x_idx=range(0,num_pix_side)
y_idx=range(0,num_pix_side)


#Random Locations of grid cells on a Grid
x_pref = np.zeros((N_in, 2))

X_pref = np.linspace(-sidelength / 2, sidelength / 2,15)
Y_pref = np.linspace(-sidelength / 2, sidelength / 2,15)
    
x_pref, y_pref = np.meshgrid(X_pref,Y_pref)

x_pref=np.reshape(x_pref, N_in)
y_pref=np.reshape(y_pref,N_in)

x_pref=np.c_[x_pref,y_pref]

Locations=np.zeros((t_max/dt,2))

Weights=[]

#Give Grid Cell Locations to the GridCell Class
GridCell=Grid_cell.GridCell()
GridCell.set_Grid_Locations(x_pref)
#Initialize rplus vector
r_plus=np.zeros((N_out,t_max/dt))
r_minus=np.zeros((N_out,t_max/dt))


#Initilize weights
GridCell.set_initial_weights()

i=0
for t in np.arange(t_0,t_max,dt):
    
    # Get the rat's current location and store them
    GridCell.get_Rat_Location()
    Locations[i,:]= GridCell.get_Rat_Location()
    
    # Gett current learning parameters
    r_plus[:,i]=GridCell.get_rplus()
    r_minus[:,i]=GridCell.get_rminus()

    #Update 
    GridCell.update()

    # Get feed forward weights
    Weights=(GridCell.get_weights())
    
    i+=1
    #Plot and/or save weight distribution for every 10000th step
    if i%10000 == 0:
        activations=np.zeros((num_pix_side,num_pix_side))
        for x in x_idx:
            for y in y_idx:
                location=np.array((x,y))*delta_r+np.array((-sidelength/2,-sidelength/2))
                rate=np.zeros(N_in)
                for k in range(np.shape(x_pref)[0]):
                    rate[k] = np.exp(-((np.linalg.norm(location - x_pref[k,:])) ** 2) / (2 * (sig_p ** 2)))        
                h=np.dot(Weights[:,Unit],rate)
                activations[x,y]=h
                
        # Displaying current weight distribution        
        # plt.imshow(activations,vmin=-0.2, vmax=0.6)#
    
        # Saving images
        plt.savefig(r'grid%i.png' %i)
    

########################### Plotting #######################
# Plot random start Location
plt.scatter(x_pref[:,0],x_pref[:,1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Random start positions of grid cells')
plt.show()


plt.plot(np.arange(t_0,t_max,dt), r_plus[10,:])
plt.title(r'$r_{plus}$ Evolution')
plt.xlabel('time t')
plt.ylabel(r'$r_{plus}$')
plt.show()

plt.figure(figsize=(10,10))
plt.plot(Locations[:,0], Locations[:,1])
plt.xlabel('x location')
plt.ylabel('y location')
plt.title('Rat Locations')
plt.show()


