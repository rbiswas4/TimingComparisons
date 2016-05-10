import timeit
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numba
sns.set()



# using setitem
t = timeit.Timer("print 'main statement'", "print 'setup'")

print 'TIMEIT:'
print t.timeit(2)

print 'REPEAT:'
print t.repeat(3, 2)

print "DONE"

@numba.jit(nopython=False)
def sumvars(x):
    res = 0.
    for e in x:
        res += e
    return res
res = [] 
yres = [] 
for n in np.logspace(1, 6, 15):
    num = str(n)
    setup = "import numpy as np; n = " + num + " ; x=np.random.random(n)"
    ysetup = setup + "; from __main__ import sumvars "
    t = timeit.Timer("np.sum(x)", setup)
    y = timeit.Timer("sumvars(x)", ysetup)
    # t = timeit.Timer("np.sum(x)", "import numpy as np; n = 50000000; x=np.random.random(n)")
    vals = t.repeat (3, 2000)
    yvals = y.repeat (3, 2000)
    res.append([n] + list(vals))
    yres.append([n] + list(yvals))
    print (n)
print(res)
res = np.asarray(res)
yres = np.asarray(yres)
plt.errorbar(res[:, 0], res[:, 1:].mean(axis=1) , yerr=res[:, 1:].std(axis=1), fmt='o')
plt.errorbar(yres[:, 0], yres[:, 1:].mean(axis=1) , yerr=yres[:, 1:].std(axis=1), fmt='s')
plt.savefig('fig2.png')
plt.show()
