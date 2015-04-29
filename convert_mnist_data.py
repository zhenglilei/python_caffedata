# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:07:09 2015

@author: root
"""

import lmdb
import leveldb
import struct
import os
import numpy as np

# Make sure that caffe is on the python path:
caffe_root = '/home/liris/Downloads/caffe/'  
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe


def main():
    #---------------------- configuration ---------------------   
    db_backend = 'leveldb'    
    image_filename = caffe_root + 'data/mnist/t10k-images-idx3-ubyte'
    label_filename = caffe_root + 'data/mnist/t10k-labels-idx1-ubyte'
    
    #---------------------- reading header --------------------
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
    
    #---------------- readin data and write out database-----------
    print ('Opening db')    
    dp_path = 'mnist_test_' + db_backend + '_py'
    if not os.path.exists(dp_path):    
        os.mkdir(dp_path)
    
    if db_backend == 'leveldb':
        db = leveldb.LevelDB(dp_path)
        batch = leveldb.WriteBatch()
    elif db_backend == 'lmdb':
        mdb_env = lmdb.open(dp_path, map_size=int(1e12)) 
        mdb_txn = mdb_env.begin(write=True)
    else:
        raise Exception('Unknown db backend')
    
    datum = caffe.proto.caffe_pb2.Datum()
    datum.channels = 1    
    datum.height = rows
    datum.width = cols
        
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
        
        if db_backend == 'leveldb':
            batch.Put( keystr, datum.SerializeToString() )
        elif db_backend == 'lmdb':
            mdb_txn.put( keystr, datum.SerializeToString() )
        else:
            raise Exception('Unknown db backend')
        
        
        if (item_id+1)%1000 == 0: # write down the buffer every 1000 data
            print ( str(item_id+1) + ' data passed')
            if db_backend == 'leveldb':
                db.Write(batch, sync=True)
                batch = leveldb.WriteBatch()
            elif db_backend == 'lmdb':
                mdb_txn.commit()
                mdb_txn = mdb_env.begin(write=True)
            else:
                raise Exception('Unknown db backend')
           
    print('-----')        
            
    
    if db_backend == 'leveldb':
        if (item_id+1)%1000 != 0: # write down the last part if there is
            db.Write(batch, sync=True)
    elif db_backend == 'lmdb':
        if (item_id+1)%1000 != 0: # write down the last part if there is
            mdb_txn.commit()
            mdb_txn = mdb_env.begin(write=True)
        mdb_env.close()
    else:
        raise Exception('Unknown db backend')        

    
if __name__ == '__main__':
    main()