# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 11:02:14 2015

@author: Oliver
"""

import numpy as np

class GridCell:
    
    def __init__(self):
        # setting class parameters
        self.params = {\
        'sidelength': 1.25,\
        'angle': 0.0,\
        'RatPos': np.array([0.0, 0.0]),\
        'GridPos': np.zeros((225, 2)),\
        'speed': 0.4,\
        't': 0.0,\
        't_max': 500,\
        'dt': .01,\
        'Rates': np.zeros((225)),\
        'sigp': 0.05,\
        'h': 0.0,\
        'rplus': np.zeros(100),\
        'rminus': np.zeros(100),\
        'tauplus': 0.1,\
        'tauminus': 0.3,\
        'hi': np.zeros(100),\
        'N_out': 100,\
        'a': 0.1,\
        's':0.3,\
        }
        self.t=0.0
        self.weights=np.zeros((225,100))
        self.rout=np.zeros(100)
        self.rplusgrid=np.zeros(100)
        self.rout_mean=np.zeros(100)
        self.Rates_mean=np.zeros(225)
        self.gain=4.5
        self.threshold=0.0
        self.gainvalues=[]
        self.thresholdvalues=[]
        
    def heaviside(self,the_x):
        if the_x > 0:
            the_result = 1
       # elif the_x == 0:
        #    the_result = 0.5
        else:
            the_result = 0
    
        return the_result
    
        
        
    def Rat_Location(self, params):
        RatPos=self.params['RatPos']
        speed=self.params['speed']*self.params['dt']
        sidelength=self.params['sidelength']
        angle=self.params['angle']
        
        angle = np.random.normal(angle, 0.2)        
        RatPos = RatPos + ([np.sin(angle), np.cos(angle)] / (np.linalg.norm([np.sin(angle), np.cos(angle)]))) * speed
        self.params['RatPos']=RatPos
        
        while (abs(RatPos[0]) > sidelength / 2.0) or (abs(RatPos[1]) > sidelength / 2.0):
            RatPos=self.params['RatPos']
            angle = np.random.normal(angle, 0.2)
            RatPos = RatPos + ([np.sin(angle), np.cos(angle)] / (np.linalg.norm([np.sin(angle), np.cos(angle)]))) * speed
        

        return RatPos, angle
        
    def get_Rat_Location(self):
        return self.params['RatPos']            
        
    def eulerstep(self,f_func, startvalue, params):
        return startvalue + f_func(startvalue, self.params) * params['dt']
    
    def set_Grid_Locations(self, Locations):
        self.params['GridPos']=Locations

    def firingrate(self,  params):
        GridPos=self.params['GridPos']
        RatPos=self.params['RatPos']
        sig_p=self.params['sigp']
        Rates=self.params['Rates']
        
        for i in range(np.shape(GridPos)[0]):
            Rates[i] = np.exp(-((np.linalg.norm(RatPos - GridPos[i,:])) ** 2) / (2 * (sig_p ** 2)))
        return Rates
        
    def get_firingrate(self):
        return self.params['Rates']
        
        
        #Adaptation Dynamics
    def rminusdot(self, rminus, params):
        tauminus=self.params['tauminus']
        #h=self.hfunc(self.params)
        h=self.params['hi']
        return (h - rminus) / tauminus
        
    def rplusdot(self, rplus, params):
        tauplus=params['tauplus']
        rminus=params['rminus']
       # h=params['h']
        
        rminus=self.eulerstep(self.rminusdot, rminus, self.params)
        self.params['rminus']=rminus
       # h=self.hfunc(self.params)
        h=self.params['hi']
        return (h - rplus - rminus) / tauplus
    
    def hfunc(self, params):
        k=0.1
        t=self.t
        
        return np.sin(np.pi * k * (t ** 2))
        
    def get_rplus(self):
        #return self.params['rplus']
        return self.rplusgrid
        
    def get_rminus(self):
        return self.params['rminus']

        
    #Initialize Weights
    def randweights(self):
        eta=0.1
        #eta=0.2        
        weights=np.zeros(225)
        for i in range(len(weights)):
            weights[i] = (1 - eta) + eta * (np.random.uniform(0, 1))
        weights = weights / (np.linalg.norm(weights))
    #    self.params['weights']=weights
        return np.transpose(weights)
        
    def set_initial_weights(self):
        self.weights=np.zeros((225,100))
        for i in range(100):
            self.weights[:,i]=self.randweights() 
        return self.weights
        
  #  def get_weights(self):
   #     return self.params['weights']
        
    #Feed forward input
    def h_i(self):
        h=np.zeros(100)
        for i in range(100):
            h[i]=np.dot(self.weights[:,i], self.params['Rates'])
        return h
        
    def get_h_i(self):
        return self.params['hi']
        
    #Adaptation Dynamics
    def OutputRate(self):
        rplus=self.rplusgrid
        threshold=self.threshold
        gain=self.gain
        
        rout=np.zeros(100)
        
        for i in range(len(rout)):
            rout[i]= (2 / np.pi) * np.arctan(gain * (rplus[i] - threshold)) * self.heaviside(rplus[i] - threshold)
        
        return rout
        
    def get_OutputRate(self):
        return self.rout
      
    def fatigue_dynamics(self, params):
        limit=100
        N_out=self.params['N_out']
        r_out=self.rout
        
        gain=self.gain
        threshold=self.threshold
        
        a = np.sum(r_out) / float(N_out)
        s = ((np.sum(r_out))**2) / (float(N_out)*(np.sum(r_out**2)))        
        self.params['a'], self.params['s'] = a, s
       # print 's outer loop', s
        #print 'a outer loop', a

        #gain=gain+0.1*gain*(s-0.3)
        #threshold=threshold+0.01*(a-0.1)
        self.gain=gain
        self.threshold=threshold
        
        
        
        if self.t <= limit*self.params['dt']:
            
            while (abs(a-0.1)/0.1) > 0.1:
                threshold=self.threshold
                threshold=threshold+0.01*(a-0.1)
                
                self.threshold=threshold
                
                r_out=self.OutputRate()
                a = np.sum(r_out) / float(N_out)
              #  A.append(a)
               # print A
                self.params['a']=a
               

        i=0
        if self.t > limit*self.params['dt']:
            
            while  (abs(a-0.1)/0.1) > 0.1 or (abs(s-0.3)/0.3) >0.1:

                if (abs(a-0.1)/0.1) > 0.1:
                    threshold=self.threshold
                    threshold=threshold+0.01*(a-0.1)
                    
                    self.threshold=threshold
                    
                    r_out=self.OutputRate()
              
                    
                
                if  (abs(s-0.3)/0.3)  > 0.1:
                    gain=self.gain
                    gain=gain+0.1*gain*(s-0.3)
                    self.gain=gain
                    
                r_out=self.OutputRate()
                a = np.sum(r_out) / float(N_out)
                s = ((np.sum(r_out))**2) / (float(N_out)*(np.sum(r_out**2)))        

                self.params['s']=s
                self.params['a']=a
                i+=1
                if i > 100:
                    if i%500 == 0:
                        print i, 'time:', self.t, self.params['s']
                if i > 200:
                    break


        
        return  gain, threshold
        
    def learning_dynamics(self):
        eps=0.005
        #eps=0.02

        weights=self.weights
        rout_mean=self.rout_mean
        Rates_mean=self.Rates_mean
        
        for i in range(100):
            weights[:,i]=weights[:,i]+eps*(self.rout[i]*self.params['Rates']-rout_mean[i]*Rates_mean)
            weights[:,i] = weights[:,i] / (np.linalg.norm(weights[:,i]))

        return weights
        
    def get_weights(self):
        return self.weights
        
    def estimate_mean_rates(self):
        rout_mean=self.rout_mean
        Rates_mean=self.Rates_mean
        rout=self.rout
        Rates=self.params['Rates']
        eta=0.05
        
        rout_mean=rout_mean + eta*(rout-rout_mean)
        Rates_mean=Rates_mean + eta*(Rates-Rates_mean)
        
        return rout_mean, Rates_mean
        
    def get_mean_rates(self):
        return self.rout_mean, self.Rates_mean
        
#    def get_gain_values(self):
#        return self.gainvalues, self.threshold
        
    def get_gain_threshold(self):
        return self.gain, self.threshold
        
    def get_a_s(self):
        return  self.params['a'], self.params['s']
             
    def update(self):
        self.t+= self.params['dt']

        self.params['RatPos'], self.params['angle']= self.Rat_Location(self.params)
        self.params['Rates']=self.firingrate(self.params)
    
        self.rplusgrid=self.eulerstep(self.rplusdot,self.rplusgrid,self.params)
        
        self.params['hi']=self.h_i()
        
        
        self.rout=self.OutputRate()
        

        #Update fatigue dynamics
        self.gain, self.threshold = self.fatigue_dynamics(self.params)
       # self.params['gain'], self.params['threshold'] = 0.1, 0.3
        
         #Update estimated mean firing rates
        self.rout_mean, self.Rates_mean = self.estimate_mean_rates()

        #Update weights
        self.weights=self.learning_dynamics()
        

    
        