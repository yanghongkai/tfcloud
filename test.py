#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
import time
import datetime

start=datetime.datetime.now()
sum=0
for i in xrange(10000000):
    sum+=1
end=datetime.datetime.now()
span=end-start
print(span)
#print(end.seconds)






