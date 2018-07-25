# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 23:07:27 2017

@author: prate
"""

import sys
# append tinyfacerec to module search path
sys . path . append ("..")
# import numpy and matplotlib colormaps
import numpy as np
# import tinyfacerec modules
from tinyfacerec . subspace import pca
from tinyfacerec . util import normalize , asRowMatrix , read_images
from tinyfacerec . visual import subplot

def read_images ( path , sz = None ):
    c = 0
    X ,y = [] , []
    for dirname , dirnames , filenames in os . walk ( path ):
        for subdirname in dirnames :
            subject_path = os . path . join ( dirname , subdirname )
            for filename in os . listdir ( subject_path ):
                try :
                    im = Image . open ( os . path . join ( subject_path , filename ))
                    im = im . convert ("L")
                    # resize to given size (if given )
                    if ( sz is not None ) :
                        im = im . resize (sz , Image . ANTIALIAS )
                        X. append ( np . asarray (im , dtype = np . uint8 ) )
                        y. append (c)
                except IOError :
                    print( "I/O error ({0}) : {1} ". format ( errno , strerror ))
                except :
                    print (" Unexpected error :", sys . exc_info () [0])
                    raise
            c = c +1
    return [X , y]



def asRowMatrix (X) :
    if len (X) == 0:
     return np . array ([])
mat = np . empty ((0 , X [0]. size ) , dtype = X [0]. dtype )
for row in X:
mat = np . vstack (( mat , np . asarray ( row ). reshape (1 , -1) ))
return mat
def asColumnMatrix (X):
if len (X) == 0:
return np . array ([])
mat = np . empty (( X [0]. size , 0) , dtype = X [0]. dtype )
for col in X:
mat = np . hstack (( mat , np . asarray ( col ). reshape ( -1 ,1) ))
return mat




def pca (X , y , num_components =0) :
[n , d] = X . shape
if ( num_components <= 0) or ( num_components >n) :
num_components = n
mu = X. mean ( axis =0)
X = X - mu
if n > d:
C = np . dot (X.T ,X)
[ eigenvalues , eigenvectors ] = np . linalg . eigh (C)
else :
C = np . dot (X ,X .T)
[ eigenvalues , eigenvectors ] = np . linalg . eigh (C)
eigenvectors = np . dot (X .T , eigenvectors )
for i in xrange (n):
eigenvectors [: , i ] = eigenvectors [: , i ]/ np . linalg . norm ( eigenvectors [: , i ])
# or simply perform an economy size decomposition
# eigenvectors , eigenvalues , variance = np. linalg . svd (X.T, full_matrices = False )
# sort eigenvectors descending by their eigenvalue
idx = np . argsort ( - eigenvalues )
eigenvalues = eigenvalues [ idx ]
eigenvectors = eigenvectors [: , idx ]
# select only num_components
eigenvalues = eigenvalues [0: num_components ]. copy ()
eigenvectors = eigenvectors [: ,0: num_components ]. copy ()
return [ eigenvalues , eigenvectors , mu ]


def project (W , X , mu = None ):
if mu is None :
return np . dot (X ,W)
return np . dot (X - mu , W)


def reconstruct (W , Y , mu = None ) :
if mu is None :
return np . dot (Y ,W.T)
return np . dot (Y , W .T) + mu



# read images
[X , y] = read_images ('D:\homework and assignments\computer vision\face and gender recognition\images')
# perform a full pca
[D , W , mu ] = pca ( asRowMatrix (X ) , y)