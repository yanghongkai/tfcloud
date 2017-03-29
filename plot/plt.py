#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
import matplotlib.pyplot as plt
import codecs
import re
import yfile
import os


def copy_file(filename):
    f=codecs.open(filename,"r",encoding="utf-8")
    xlist=[]
    ylist=[]
    for line in f:
        line=line.strip()
        arr=re.split(r'[;,\s]\s*',line)
        xlist.append(arr[0])
        ylist.append(arr[2])
        #ylist.append(arr[1])
    xlist_=[int(x)/1000 for x in xlist]

    return xlist_,ylist

def main():
    rootdir='/home/yhk/gitproject/tfcloud/plot/draw/'
    jpgname=""
    filenames=yfile.getFileDir(rootdir)
    filenames_=[filename.split('/')[-1] for filename in filenames]
    #print(filenames_)
    vx=None
    vy=None
    tx=None
    ty=None
    for filename in filenames_:
        if re.match(r'valid',filename):
            jpgname=os.path.join(rootdir,filename+".jpg")
            filepath=os.path.join(rootdir,filename)
            print(filepath)
            vx,vy=copy_file(filepath)
        else:
            filepath=os.path.join(rootdir,filename)
            tx,ty=copy_file(filepath)
    ax=plt.subplot(1,1,1)
    vline=ax.plot(vx,vy,'bo-',label='valid')
    tline=ax.plot(tx,ty,'r+-',label='test')
    plt.axis([0,len(vx),0,100])
    #plt.title("valid and test accuracy")
    plt.title("learning_rate_base=0.05,rate_decay=0.99")
    plt.xlabel('train steps (1000)')
    plt.ylabel('accuracy %')
    handles,labels=ax.get_legend_handles_labels()
    ax.legend(handles[::1],labels[::1])
    #plt.show()
    plt.savefig(jpgname)
        
            

    

if __name__=='__main__':
    main()









