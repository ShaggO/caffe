import numpy as np
import caffe
def list2blobproto( filename, diagonal ):
  L = len(diagonal)
  H = np.eye( L, dtype = 'f4' )
  np.fill_diagonal( H, diagonal )
  blob = caffe.io.array_to_blobproto( H.reshape( (1,1,L,L) ) )
  with open( filename, 'wb' ) as f :
    f.write( blob.SerializeToString() )