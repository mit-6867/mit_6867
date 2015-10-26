from numpy import *
from plotBoundary import *
import scipy.optimize
import matplotlib as mpl
from ggplot import *
import pandas as pd

# Logistic Regression Loss Function
def LRLoss(w, X, Y, l=0):
    return sum(log(1+exp(-hstack(Y)*(dot(X,w[1:len(w)])+w[0]))))+l*dot(w[1:len(w)], w[1:len(w)])

# Prediction Function
def LRPredict(w, X):
    return 1/(1+exp(-(dot(X,w[1:len(w)])+w[0])))

# Data Loading Function
def loadData(name, typ):
    try:
        data = loadtxt('/Users/dholtz/Downloads/hw2_resources/data/data_'+name+'_'+typ+'.csv')
    except:
        data = loadtxt('/Users/mfzhao/Downloads/hw2_resources/data/data_'+name+'_'+typ+'.csv')
    X = data[:,0:2]
    Y = data[:,2:3]
    return X, Y

# Training Function
def Train(X, Y, l=0):
    w0 = random.randn(3)
    w_opt = scipy.optimize.fmin_bfgs(LRLoss, w0, args=(X,Y,l))
    return w_opt

# Classification Error Function
def classifyErr(py, Y, db):
    yhat = ones(len(py))
    for i in range(len(py)):
        if py[i] < db:
            yhat[i] = -1
    return 1-mean(Y.flatten() == yhat)

# Classification Error vs Decision Boundary Plotting
def plotCEDB(w, X, Y, title):
    global pltsize
    axis = array(linspace(0,1,101))
    out = zeros(len(axis))
    py = LRPredict(w,X)
    for i in range(len(axis)):
        out[i] = classifyErr(py, Y, axis[i])
    DF = pd.DataFrame({'Decision Boundary': pd.Series(axis),'Classification Error': pd.Series(out)})
    print ggplot(DF, aes(x='Decision Boundary', y='Classification Error')) + geom_line(size=4) + ggtitle(title) + theme_matplotlib(rc=pltsize, matplotlib_defaults=False) 

# Grid Search Lambda Function
def GridL(Xt, Yt, Xv, Yv, l, db=0.5):
    tErr = []
    vErr = []
    tClass = []
    vClass = []
    w0 = random.randn(3)
    for i in range(len(l)):
        w = scipy.optimize.fmin_bfgs(LRLoss, w0, args=(Xt,Yt,l[i]))
        tErr.append(LRLoss(w, Xt, Yt, 0)/len(Yt))
        tClass.append(classifyErr(LRPredict(w, Xt), Yt, db))
        vErr.append(LRLoss(w, Xv, Yv, 0)/len(Yv))
        vClass.append(classifyErr(LRPredict(w, Xv), Yv, db))
    return array(tErr), array(tClass), array(vErr), array(vClass)

# Wrapper to do everything necessary for each dataset
def wrapper(name):
    Xt, Yt=loadData(name, 'train')
    Xv, Yv=loadData(name, 'validate')
    w = Train(Xt, Yt, 0)
    print 'Classification Error (TR): ', classifyErr(LRPredict(w, Xt), Yt, 0.5), name
    print 'Classification Error (VAL):: ',classifyErr(LRPredict(w, Xv), Yv, 0.5), name
    t1 = 'Classification Error vs Decision Boundary - ' + name + ': Training'
    t2 = 'Classification Error vs Decision Boundary - ' + name + ': Validation'
    plotCEDB(w, Xt, Yt, '')
    plotCEDB(w, Xv, Yv, '')
    t1 = 'Logistic Regression - ' + name + ': Training'
    t2 = 'Logistic Regression - ' + name + ': Validation'
    plotDecisionBoundary(w, Xt, Yt, LRPredict, [0.5], '')
    plotDecisionBoundary(w, Xv, Yv, LRPredict, [0.5], '')
    l = array(linspace(0,100,101))
    tErr, tClass, vErr, vClass = GridL(Xt, Yt, Xv, Yv, l)
    DF1 = pd.DataFrame({'TR': pd.Series(tClass), 'VAL': pd.Series(vClass), 'Lambda': pd.Series(l)})
    DF1 = pd.melt(DF1,id_vars=['Lambda'])
    DF2 = pd.DataFrame({'TR': pd.Series(tErr), 'VAL': pd.Series(vErr), 'Lambda': pd.Series(l)})
    DF2 = pd.melt(DF2,id_vars=['Lambda'])
    title1 = 'Classification Error vs Lambda - ' + name
    title2 = 'Logisitic Loss vs Lambda - ' + name
    #print p1 = ggplot(DF1, aes(x='Lambda', y='value', color='variable')) + geom_line(size=4) + ggtitle('') + ylab('Error')
    #print p2 = ggplot(DF2, aes(x='Lambda', y='value', color='variable')) + geom_line(size=4) + ggtitle('') + ylab('Error')

def wrapper2(name):
    global pltsize
    Xt, Yt=loadData(name, 'train')
    Xv, Yv=loadData(name, 'validate')
    w = Train(Xt, Yt, 0)
    return w

w1 = wrapper2('stdev1')
w2 = wrapper2('stdev2')
w3 = wrapper2('stdev4')
w4 = wrapper2('nonsep')

DF = pd.DataFrame({'stdev1': pd.Series(w1), 'stdev4': pd.Series(w1), 'stdev2': pd.Series(w1), 'nonsep': pd.Series(w4)})
DF.to_latex()


