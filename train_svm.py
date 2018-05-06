import itertools
import os
import os.path as osp
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

from spectral.io.bsqfile import BsqFile
from spectral.io.envi import *
import numpy as np
#import cv2
from spectral import *

def rp(scores, gt):
    scores=scores.reshape(-1);
    gt=gt.reshape(-1);
    print('R/P:',metrics.average_precision_score(gt, scores))


class Process:
    def __init__(self, train_data):
        train_data=np.array(train_data, dtype='f8')
        self.mean=train_data.mean(axis=(0,1))
        X=self.raster(train_data)
        X-=self.mean
        if np.abs(X.mean()) > 1e-13:
            import ipdb; ipdb.set_trace()
            raise Exception('Wrong rasterization.')
    def raster(self, X):
        #X=np.array([[X[k,l,:] for k in range(X.shape[0])] for l in range(X.shape[1])]).reshape(-1, X.shape[-1])
        X=X.reshape(-1, X.shape[-1])
        return X
    def normalize(self, X):
        X=np.array(X, 'f')
        X=np.array(X)
        X-=self.mean
        X/=np.linalg.norm(X, axis=1).reshape([-1,1])
        return X
    def preproc(self, X):
        return self.normalize(self.raster(X))
        

def train_test(X, Y, split_col, tag):
    #X=np.array([X[:,:,k] for k in range(X.shape[-1])], dtype='f')
    output_dir='results'
    try: os.mkdir(output_dir)
    except: pass
    
    train_data  =X[:,split_col:,:]
    proc=Process(train_data)
    train_data=proc.preproc(train_data)
    train_labels=Y[:,split_col:].reshape(-1)

    # #DBG
    # train_data=train_data[:10]
    # train_labels=train_labels[:10]
    # train_labels[0]=1
    
    test_data =proc.preproc(X[:,:split_col,:])
    test_labels=Y[:,:split_col].reshape(-1)

    #Train
    clf = svm.SVC()
    clf=clf.fit(train_data, train_labels)

    #Test
    pred=clf.predict(test_data)
    plt.imsave(osp.join(output_dir, tag+'.png'), pred.reshape(X.shape[0],-1))

    # Compute eval measures
    print(pred/pred.max())
    rp(pred, test_labels)


def load_rgb_apples():
    X=plt.imread('./data/rgb_apples.png')
    Y=np.load('./data/rgb_apples_labels.npy')
    Y=Y.all(axis=2)
    return X,Y,231,'rgb_apples'

def load_hyper_apples():
   
    # read image hyper spectral
    header_path='./data/SSM5x5-NIR_09580554_20160605171039.hdr'
    filename='./data/SSM5x5-NIR_09580554_20160605171039.bsq'

    h = read_envi_header(header_path)
    p = gen_params(h)
    p.filename=filename
    img_hyper = BsqFile(p, h)

    height, width, channels = img_hyper.shape
    img_hyper = img_hyper[:,:,:]
    img_hyper=np.array(img_hyper, dtype='f')
    img_hyper/=img_hyper.max()
    #import ipdb; ipdb.set_trace()
    print(img_hyper.shape)

    # read hyperspectral mask
    path_hyper_mask="./data/hyper_apples_labels.npy"
    mask_hyper=np.load(path_hyper_mask)
    print(mask_hyper.shape)

    #
    return img_hyper, mask_hyper, 197, 'hyper_apples'


    
if __name__=='__main__':
    
    # X=np.random.rand(10,10, 3)
    # Y=np.random.randint(2,size=(10,10))
    # train_test(X, Y, 5)

    #X,Y,N,tag=load_rgb_apples()
    X,Y,N,tag=load_hyper_apples()
    train_test(X, Y, N, tag)

