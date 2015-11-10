from hw3mk2 import *
import scipy

pathTr = "/Downloads/hw3_resources/toy_multiclass_1_train.csv"
pathV = "/Downloads/hw3_resources/toy_multiclass_1_validate.csv"
pathTe = "/Downloads/hw3_resources/toy_multiclass_1_test.csv"
try:
    filename = "/Users/mfzhao/"+pathTr
    Tr = scipy.io.loadmat(filename, appendmat=False)['toy_data']
    filename = "/Users/mfzhao/"+pathV
    V = scipy.io.loadmat(filename, appendmat=False)['toy_data']
    filename = "/Users/mfzhao/"+pathTe
    Te = scipy.io.loadmat(filename, appendmat=False)['toy_data']
except:
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

def gridSearch(Xtr, YtrM, Xv, YvM):
    lamb = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    h = [1,2,3,4,5,6,7,8,9,10]
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
            w1, w2, b1, b2 = NNet(Xtr, YtrM, m, l, 15, crit=1e-3)
            yhattr = nnPredict(w1, w2, b1, b2, Xtr)
            trE[i,j] = classificationError(yhattr, YtrM)
            yhatv = nnPredict(w1, w2, b1, b2, Xv)
            vE[i,j] = classificationError(yhatv, YvM)
            j += 1
            if vE[i,j] < besterr:
                besterr = vE[i,j]
                besttr = trE[i,j]
                bestl = l
                bestm = m
        i += 1
    print 'optimal lambda:       ', bestl
    print 'optimal hidden units: ', bestm
    print 'Training Error:       ', besttr
    print 'Validation Error:     ', besterr
    return trE, vE
    
def SGDgrid(Xtr, YtrM, Xv, YvM):
    lamb = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    h = [1,2,3,4,5,6,7,8,9,10]
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
            w1, w2, b1, b2 = sgdNNet(Xtr, YtrM, m, l, 15, 10, crit=1e-3)
            yhattr = nnPredict(w1, w2, b1, b2, Xtr)
            trE[i,j] = classificationError(yhattr, YtrM)
            yhatv = nnPredict(w1, w2, b1, b2, Xv)
            vE[i,j] = classificationError(yhatv, YvM)
            j += 1
            if vE[i,j] < besterr:
                besterr = vE[i,j]
                besttr = trE[i,j]
                bestl = l
                bestm = m
        i += 1
    print 'SGD - optimal lambda:       ', bestl
    print 'SGD - optimal hidden units: ', bestm
    print 'SGD - Training Error:       ', besttr
    print 'SGD - Validation Error:     ', besterr
    return trE, vE
    
TrErr_M, VErr_M = gridSearch(Xtr, YtrM, Xv, YvM)
TrErr_M, VErr_M = SGDgrid(Xtr, YtrM, Xv, YvM)