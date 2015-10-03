import numpy as np
import scipy.optimize as spo
import pandas as pd 

np.set_printoptions(precision=4)

# Generic gradient descent function
def gradientDescent(f, df, init, lr=0.3, crit=0.0001, maxIter=1000, h=0.001):
	count = 0
	nIter = 0
	fcall = 0
	while count < 2 and nIter <= maxIter:
		f_i = f(init)
		# Calculates the gradient
		grad = df(init, f, h)
		# Update step
		init = init - lr * grad
		f_u = f(init)
		# Calculate the difference in f using initial and updated values
		diff = abs((f_i - f_u))
		nIter += 1
		fcall += 2
		if df == centdiff:
			fcall += 2*len(init)
		# Tracks successive number of times that difference is below the convergence criterion
		if diff < crit:
			count += 1
		else:
			count = 0

	print "nIter: %d" % (nIter)
	print "Fcall: %d" % (fcall)
	return init

# Central difference approximation of a gradient, returns a vector of length of the input vector x
def centdiff(x, f, h=0.00001):
	n = len(x)
	out = np.zeros(n)
	for i in range(0, n):
		hplus = np.copy(x)
		hminus = np.copy(x)
		hplus[i] += h
		hminus[i] -= h
		# Calculates a better denominator to address potential problems with floating point arithmetic especially for small values of h
		hfix = hplus[i] - hminus[i]
		out[i] = (f(hplus) - f(hminus))/(hfix)
	return out

# Quadratic bowl in n-dimensions, n determined by length of input vector x, optimal value at [0, 0, ..., 0]
def f1(x):
	return np.dot(x, x)

def df1(x, *args):
	return 2*x

# Non-convex function with multiple local minima, global minimum at [0, 0, ..., 0]
def f2(x):
	return sum(x**2/100 - np.cos(x))

def df2(x, *args):
	return (x/50 + np.sin(x))

# Various starting values
start1 = np.array([5., -3., 7., 8.])
start2 = np.array([-10., 4., 1., -5.])
start3 = np.array([1., -1., -3., 2.])

results = np.empty([1,5])

for i in [[5., -3., 7., 8.], [-10., 4., 1., -5.], [1., -1., -3., 2.]]:
	for lr in [.3, .03, .003]:
		for k in [.1, .001, .00001]:
			results = np.vstack([results, np.array([i, lr, k, ['%.3f' % elem for elem in gradientDescent(f1, df1, np.array(i), lr=lr, crit=k)], ['%.3f' % elem for elem in gradientDescent(f2, df2, np.array(i), lr=lr, crit=k)]]).reshape(1,5)])

resultsDF = pd.DataFrame(results)

print resultsDF[1:].to_latex(index_names = False)

# 2 - Testing gradient descent procedure
print "-----------Gradient Descent Procedure Testing-----------\n"
print "Function 1: Quadratic Bowl, Global Min at (0, 0, ..., 0)\n"

print "Start:",np.array_str(start1)
f1s1 = gradientDescent(f1, centdiff, start1)
print "  End:",np.array_str(f1s1),"\n"

print "Start:",np.array_str(start2)
f1s2 = gradientDescent(f1, centdiff, start2)
print "  End:",np.array_str(f1s2),"\n"

print "Start:",np.array_str(start3)
f1s3 = gradientDescent(f1, centdiff, start3)
print "  End:",np.array_str(f1s3),"\n"

print "Function 2: Nonconvex Multiple Local Min, Global Min at (0, 0, ..., 0)\n"

print "Start:",np.array_str(start1)
f2s1 = gradientDescent(f2, centdiff, start1)
print "  End:",np.array_str(f2s1),"\n"

print "Start:",np.array_str(start2)
f2s2 = gradientDescent(f2, centdiff, start2)
print "  End:",np.array_str(f2s2),"\n"

print "Start:",np.array_str(start3)
f2s3 = gradientDescent(f2, centdiff, start3)
print "  End:",np.array_str(f2s3),"\n"
print "--------------------------------------------------------\n\n"

# 3 - Testing Central Differences approximation
print "--------Testing Central Difference Approximation--------"
print "Function 1, Values:",start1
print "Analytical Grad:",df1(start1)
print " Numerical Grad:",centdiff(start1, f1),"\n"

print "Function 1, Values:",start2
print "Analytical Grad:",df1(start2)
print " Numerical Grad:",centdiff(start2, f1),"\n"

print "Function 1, Values:",start3
print "Analytical Grad:",df1(start3)
print " Numerical Grad:",centdiff(start3, f1),"\n"

print "Function 2, Values:",start1
print "Analytical Grad:",df2(start1)
print " Numerical Grad:",centdiff(start1, f2),"\n"

print "Function 2, Values:",start2
print "Analytical Grad:",df2(start2)
print " Numerical Grad:",centdiff(start2, f2),"\n"

print "Function 2, Values:",start3
print "Analytical Grad:",df2(start3)
print " Numerical Grad:",centdiff(start3, f2),"\n"
print "--------------------------------------------------------\n"

# 4 - Comparison to scipy.optimize
print "--------------Comparison to scipy.optimize--------------"
print "Function 1, Method: BFGS, start:",start1
spo.fmin_bfgs(f1, start1)
print "\n"

print "Function 1, Method: CG, start:",start1
spo.fmin_cg(f1, start1)
print "\n"

print "Function 2, Method: BFGS, start:",start1
spo.fmin_bfgs(f2, start1)
print "\n"

print "Function 2, Method: CG, start:",start1
spo.fmin_cg(f2, start1)
print "\n"
print "--------------------------------------------------------"
