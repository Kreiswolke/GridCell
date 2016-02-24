# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 11:24:17 2015

@author: Oliver
"""

def heaviside(the_x):
    if the_x > 0:
        the_result = 1
    elif the_x == 0:
        the_result = 0.5
    else:
        the_result = 0

    return the_result
