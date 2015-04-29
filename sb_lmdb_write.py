# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:07:09 2015

@author: root
"""

import lmdb
import struct
import os
import numpy as np

# Make sure that caffe is on the python path:
caffe_root = '/home/liris/Downloads/caffe/'  
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe


def main():
    image_filename = caffe_root + 'data/mnist/t10k-images-idx3-ubyte'
    label_filename = caffe_root + 'data/mnist/t10k-labels-idx1-ubyte'
    
    image_file = open(image_filename, 'rb')
    magic, = struct.unpack('>i', image_file.read(4))
    if magic != 2051:
        raise ValueError('Incorrect image file magic. It should be 2015.')     
    
    label_file = open(label_filename, 'rb')    
    magic, = struct.unpack('>i', label_file.read(4))
    if magic != 2049:
        raise ValueError('Incorrect image file magic. It should be 2049')     
        
    num_items, = struct.unpack('>i', image_file.read(4))
    num_labels, = struct.unpack('>i', label_file.read(4))
    if num_items != num_labels:
        raise Exception('Unmatched numbers.')
    
    rows, = struct.unpack('>i', image_file.read(4))
    cols, = struct.unpack('>i', image_file.read(4))
    
    print (rows,cols)
    print ('Opening lmdb')    
    
    dp_path = 'mnist_test_lmdb_py'
    if not os.path.exists(dp_path):    
        os.mkdir(dp_path)
    mdb_env = lmdb.open(dp_path, map_size=int(1e12))    
    
    datum = caffe.proto.caffe_pb2.Datum()
    datum.channels = 1    
    datum.height = rows
    datum.width = cols
    
    mdb_txn = mdb_env.begin(write=True)
    
    for item_id in range(num_items):
#        print(item_id)
        strfmt = str(rows*cols)+'B'
        pixels = struct.unpack(strfmt, image_file.read(rows*cols))
        pixels = np.asarray(pixels)
        label, = struct.unpack('B', label_file.read(1))
        
        image = np.zeros((datum.channels, datum.height, datum.width))
        image[0,:,:] = pixels.reshape(rows,cols)            
        
        datum = caffe.io.array_to_datum(image,label)            
        keystr = '{:0>8d}'.format(item_id)
        mdb_txn.put( keystr, datum.SerializeToString())
        
        if (item_id+1)%1000 == 0: # write down the buffer every 1000 data
            print ( str(item_id+1) + ' data passed')
            mdb_txn.commit()
            mdb_txn = mdb_env.begin(write=True)
    print('-----')        
            
    if (item_id+1)%1000 != 0: # write down the last part if there is
        mdb_txn.commit()
    mdb_env.close()
  
    
if __name__ == '__main__':
    main()