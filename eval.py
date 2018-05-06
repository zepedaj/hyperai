import numpy as np
from sklearn import metrics

def rp(scores, gt):
    scores=scores.reshape(-1);
    gt=gt.reshape(-1);
    print('R/P:',metrics.average_precision_score(gt, scores))
    
if __name__=='__main__':
    scores=np.random.rand(10,10)
    gt=np.random.randint(2,size=(10,10))
    rp(scores, gt)
