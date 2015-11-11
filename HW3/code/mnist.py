from hw3mk2 import *
import scipy
from tabulate import tabulate

pathTr = "Downloads/hw3_resources/mnist_train.csv"
pathV = "Downloads/hw3_resources/mnist_validate.csv"
pathTe = "Downloads/hw3_resources/mnist_test.csv"
try:
    filename = "/Users/mfzhao/"+pathTr
    Tr = scipy.io.loadmat(filename, appendmat=False)['tr']
    filename = "/Users/mfzhao/"+pathV
    V = scipy.io.loadmat(filename, appendmat=False)['va']
    filename = "/Users/mfzhao/"+pathTe
    Te = scipy.io.loadmat(filename, appendmat=False)['te']
except:
    filename = "/Users/dholtz/"+pathTr
    Tr = scipy.io.loadmat(filename, appendmat=False)['tr']
    filename = "/Users/dholtz/"+pathV
    V = scipy.io.loadmat(filename, appendmat=False)['va']
    filename = "/Users/dholtz/"+pathTe
    Te = scipy.io.loadmat(filename, appendmat=False)['te']

def gridSearch(Xtr, YtrM, Xv, YvM, a):
    lamb = [0, .001, .01, .1]
    h = [200, 250]
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
            w1, w2, b1, b2 = NNet(Xtr, YtrM, m, l, a, crit=1e-3, maxiter=5000)
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
    
def SGDgrid(Xtr, YtrM, Xv, YvM, a):
    lamb = [0, .001, .01, .1]
    h = [50, 100, 150]
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
            w1, w2, b1, b2 = sgdNNet(Xtr, YtrM, m, l, a, 25, crit=1e-3, maxiter=5000)
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

Xtr = Tr[:,0:V.shape[1]-1]
Ytr = Tr[:,Tr.shape[1]-1:Tr.shape[1]]
YtrM = oneHot(Ytr)
Xv = V[:,0:V.shape[1]-1]
Yv = V[:,V.shape[1]-1:V.shape[1]]
YvM = oneHot(Yv)
Xte = Te[:,0:Te.shape[1]-1]
Yte = Te[:,Te.shape[1]-1:Te.shape[1]]
YteM = oneHot(Yte)

np.random.seed(11210)
#a = .1
#l = 0.001
#print '------------------------------Full Batch------------------------------'
#w1, w2, b1, b2 = NNet(Xtr, YtrM, 100, l, a, 1e-3, 5000)
#yhattr = nnPredict(w1, w2, b1, b2, Xtr)
#yhatv = nnPredict(w1,w2, b1, b2, Xv)

#print '------------------------------SGD:5000------------------------------'
#batch = 20
#w1, w2, b1, b2 = sgdNNet(Xtr, YtrM, 100, l, .02, batch, 1e-3, 5000)
#sgdyhattr1 = nnPredict(w1, w2, b1, b2, Xtr)
#sgdyhatv1 = nnPredict(w1,w2, b1, b2, Xv)

#print '------------------------------SGD:10000------------------------------'
#batch = 20
#w1, w2, b1, b2 = sgdNNet(Xtr, YtrM, 100, l, .02, batch, 1e-3, 10000)
#sgdyhattr2 = nnPredict(w1, w2, b1, b2, Xtr)
#sgdyhatv2 = nnPredict(w1,w2, b1, b2, Xv)

#print classificationError(yhattr, Ytr)
#print classificationError(yhatv, Yv)
#print classificationError(sgdyhattr1, Ytr)
#print classificationError(sgdyhatv1, Yv)
#print classificationError(sgdyhattr2, Ytr)
#print classificationError(sgdyhatv2, Yv)
print '--------------------------step size: .1--------------------------' 
TrErr_M1, VErr_M1 = gridSearch(Xtr, YtrM, Xv, YvM, .1)
TrErr_SGD1, VErr_SGD1 = SGDgrid(Xtr, YtrM, Xv, YvM, .1)
print '--------------------------step size: .01--------------------------'
TrErr_M2, VErr_M2 = gridSearch(Xtr, YtrM, Xv, YvM, .01)
TrErr_SGD2, VErr_SGD2 = SGDgrid(Xtr, YtrM, Xv, YvM, .01)

print tabulate(np.round(TrErr_M1, 3), tablefmt='latex')
print tabulate(np.round(VErr_M1, 3), tablefmt='latex')
print tabulate(np.round(TrErr_SGD1, 3), tablefmt='latex')
print tabulate(np.round(VErr_SGD1, 3), tablefmt='latex')
print tabulate(np.round(TrErr_M2, 3), tablefmt='latex')
print tabulate(np.round(VErr_M2, 3), tablefmt='latex')
print tabulate(np.round(TrErr_SGD2, 3), tablefmt='latex')
print tabulate(np.round(VErr_SGD2, 3), tablefmt='latex')