#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
import matplotlib.pyplot as plt

x=[1,2,3,4]
y=[1,4,9,16]
plt.plot(x,y,'bo-')
plt.axis([0,4,0,20]) # 给出坐标轴的范围[xmin,xmax,ymin,ymax]
plt.xlabel('train steps')
plt.ylabel('accuracy')
plt.show()




