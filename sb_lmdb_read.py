# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:22:20 2015

@author: root
"""

import lmdb
import os
import numpy as np
import matplotlib as mp

# Make sure that caffe is on the python path:
caffe_root = '/home/liris/Downloads/caffe/'  
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe


def main():
       
    print ('Opening lmdb')    
    
    dp_path = 'mnist_test_lmdb_py'
    if not os.path.exists(dp_path):    
        raise Exception(dp_path + 'does not exist.')
    mdb_env = lmdb.open(dp_path, map_size=int(1e12))    
    with mdb_env.begin() as mdb_txn:
        cursor = mdb_txn.cursor()  
        cursor.get('{:0>8d}'.format(9900)) # key format 
        value = cursor.value()
        key = cursor.key()
    
        print (key)         
         
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        print(datum.channels, datum.height, datum.width)
        
        image = caffe.io.datum_to_array(datum)
        print(image)
        image = image[0,:,:]
        print(image)        
        print(np.shape(image)) 
              
        mp.pyplot.imshow(image, cmap=mp.cm.gray)
        print(datum.label)
#        mp.image.imsave('example.png', image, cmap=mp.cm.gray)
        

    
if __name__ == '__main__':
    main()