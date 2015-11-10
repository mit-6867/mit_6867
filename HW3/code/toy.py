from hw3mk2 import *
import scipy

pathTr = "/Downloads/hw3_resources/toy_multiclass_1_train.csv"
pathV = "/Downloads/hw3_resources/toy_multiclass_1_validate.csv"
pathTe = "/Downloads/hw3_resources/toy_multiclass_1_test.csv"

filename = "/Users/mfzhao/"+pathTr
Tr = scipy.io.loadmat(filename, appendmat=False)['toy_data']
filename = "/Users/mfzhao/"+pathV
V = scipy.io.loadmat(filename, appendmat=False)['toy_data']
filename = "/Users/mfzhao/"+pathTe
Te = scipy.io.loadmat(filename, appendmat=False)['toy_data']

Xtr = Tr[:,0:Tr.shape[1]-1]
Ytr = Tr[:,Tr.shape[1]-1:Tr.shape[1]]
YtrM = oneHot(Ytr)
Xv = V[:,0:V.shape[1]-1]
Yv = V[:,V.shape[1]-1:V.shape[1]]
YvM = oneHot(Yv)
Xte = Te[:,0:Te.shape[1]-1]
Yte = Te[:,Te.shape[1]-1:Te.shape[1]]
YteM = oneHot(Yte)

w1, w2, b1, b2 = NNet(Xtr, YtrM, 4, 0.001, 15, crit=1e-3)
yhattr = nnPredict(w1, w2, b1, b2, Xtr)
print yhattr
print classificationError(yhattr, Ytr)
yhatv = nnPredict(w1, w2, b1, b2, Xv)
print classificationError(yhatv, Yv)
