#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
import matplotlib.pyplot as plt

ax=plt.subplot(1,1,1)
x=[1,2,3,4]
y=[i**2 for i in x]
y2=[i+1 for i in x]
#print(y)
line1=ax.plot(x,y,'bo-',label="line 1")
line2=ax.plot(x,y2,'r+-',label="line 2")
handles,labels=ax.get_legend_handles_labels()
#print(handles)
#ax.legend(handles=[line1,line2])
ax.legend(handles[::1],labels[::1])
plt.show()












