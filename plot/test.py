#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
import re

line='asdf fjdk; afed, fjek,asdf,    foo'
arr=re.split(r'(;|,|\s)\s*',line)
print(arr)






