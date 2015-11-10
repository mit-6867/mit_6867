import hw3mk2.py

path = "/Downloads/hw3_resources/mnist_train.csv"
try:
    filename = "/Users/dholtz/"+path
    T = scipy.io.loadmat(filename, appendmat=False)['tr']
except:
    filename = "/Users/mfzhao/"+path
    T = scipy.io.loadmat(filename, appendmat=False)['tr']
    
Xtr = T[:,0:T.shape[1]-1]
Ytr = T[:,T.shape[1]-1:T.shape[1]]
YtrM = oneHot(Ytr)

np.random.seed(11210)
w1, w2, b1, b2 = NNet(Xtr, YtrM, 200, 0.001, .01, crit=1e-4)
yhat = nnPredict(w1, w2, b1, b2, Xtr)

classificationError(yhat, Ytr)