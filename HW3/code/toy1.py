from hw3mk2 import *
import scipy
from tabulate import tabulate

pathTr = "/Downloads/hw3_resources/toy_multiclass_2_train.csv"
pathV = "/Downloads/hw3_resources/toy_multiclass_2_validate.csv"
pathTe = "/Downloads/hw3_resources/toy_multiclass_2_test.csv"
try:
    filename = "/Users/mfzhao/"+pathTr
    Tr = scipy.io.loadmat(filename, appendmat=False)['toy_data']
    filename = "/Users/mfzhao/"+pathV
    V = scipy.io.loadmat(filename, appendmat=False)['toy_data']
    filename = "/Users/mfzhao/"+pathTe
    Te = scipy.io.loadmat(filename, appendmat=False)['toy_data']
except:
    filename = "/Users/dholtz/"+pathTr
    Tr = scipy.io.loadmat(filename, appendmat=False)['toy_data']
    filename = "/Users/dholtz/"+pathV
    V = scipy.io.loadmat(filename, appendmat=False)['toy_data']
    filename = "/Users/dholtz/"+pathTe
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

def gridSearch(Xtr, YtrM, Xv, YvM):
    lamb = [0, 1e-5, 1e-3, 1e-1]
    h = [1,3,5,7,9]
    besttr = 1
    besterr = 1
    bestl = 0
    bestm = 1
    trE = np.zeros((len(lamb), len(h)))
    vE = np.zeros((len(lamb), len(h)))
    i=0
    for l in lamb:
        j=0
        for m in h:
            print l, m
            w1, w2, b1, b2 = NNet(Xtr, YtrM, m, l, 15, crit=1e-3)
            yhattr = nnPredict(w1, w2, b1, b2, Xtr)
            trE[i,j] = classificationError(yhattr, Ytr)
            yhatv = nnPredict(w1, w2, b1, b2, Xv)
            vE[i,j] = classificationError(yhatv, Yv)
            if vE[i,j] < besterr:
                besterr = vE[i,j]
                besttr = trE[i,j]
                bestl = l
                bestm = m
            j += 1
        i += 1
    print 'optimal lambda:       ', bestl
    print 'optimal hidden units: ', bestm
    print 'Training Error:       ', besttr
    print 'Validation Error:     ', besterr
    return trE, vE
    
def SGDgrid(Xtr, YtrM, Xv, YvM):
    lamb = [0, 1e-5, 1e-3, 1e-1]
    h = [1,3,5,7,9]
    besttr = 1
    besterr = 1
    bestl = 0
    bestm = 1
    trE = np.zeros((len(lamb), len(h)))
    vE = np.zeros((len(lamb), len(h)))
    i=0
    for l in lamb:
        j=0
        for m in h:
            print l, m
            w1, w2, b1, b2 = sgdNNet(Xtr, YtrM, m, l, 15, 10, crit=1e-3)
            yhattr = nnPredict(w1, w2, b1, b2, Xtr)
            trE[i,j] = classificationError(yhattr, Ytr)
            yhatv = nnPredict(w1, w2, b1, b2, Xv)
            vE[i,j] = classificationError(yhatv, Yv)
            if vE[i,j] < besterr:
                besterr = vE[i,j]
                besttr = trE[i,j]
                bestl = l
                bestm = m
            j += 1
        i += 1
    print 'SGD - optimal lambda:       ', bestl
    print 'SGD - optimal hidden units: ', bestm
    print 'SGD - Training Error:       ', besttr
    print 'SGD - Validation Error:     ', besterr
    return trE, vE
    
TrErr_M, VErr_M = gridSearch(Xtr, YtrM, Xv, YvM)
TrErr_SGD, VErr_SGD = SGDgrid(Xtr, YtrM, Xv, YvM)

print tabulate(np.round(TrErr_M, 3), tablefmt='latex')
print tabulate(np.round(VErr_M, 3), tablefmt='latex')
print tabulate(np.round(TrErr_SGD, 3), tablefmt='latex')
print tabulate(np.round(VErr_SGD, 3), tablefmt='latex')