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
    rootdir='/home/yhk/gitproject/tfcloud/plot/draw_multi/'
    jpgname=""
    filenames=yfile.getFileDir(rootdir)
    filenames_=[filename.split('/')[-1] for filename in filenames]
    #print(filenames_)
    colors=['bo-','r+-','g*-','kp:','b^--']
    learning_rate=0.0
    xmax=0
    ax=plt.subplot(1,1,1)
    for filename,color in zip(filenames_,colors):
        learning_rate=filename.split('-')[-1]
        filepath=os.path.join(rootdir,filename)
        x,y=copy_file(filepath)
        xmax=len(x)
        ax.plot(x,y,color,label='learning_rate='+learning_rate)
    plt.axis([0,xmax,0,100])
    #plt.title("valid and test accuracy")
    plt.title("learning_rate_base=0.05,rate_decay=0.99")
    plt.xlabel('train steps (1000)')
    plt.ylabel('accuracy %')
    handles,labels=ax.get_legend_handles_labels()
    ax.legend(handles[::1],labels[::1],loc='lower right')
    plt.show()
    #plt.savefig(jpgname)
        
            

    

if __name__=='__main__':
    main()









