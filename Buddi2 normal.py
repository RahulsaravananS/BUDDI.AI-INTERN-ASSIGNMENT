import numpy as np
import math
import matplotlib.pyplot as plt
a = 1
X = []
Y = []
n = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
txt="The below graph represent the monte carlo simulation to estimate the  value of pi using normal distribution"

def incircle(a, b):
    if (math.sqrt((a**2) + (b**2)) <= 0.5):
        return True

def check(n):
    ans = 0
    while len(X)< n+1 and len(Y)< n+1 and len(X)==len(Y):
        x1 = np.random.normal(0,0.5)
        y1 = np.random.normal(0,0.5)
        if (x1>=-0.5 or x1<=0.5) and (y1>=-0.5 or y1<=0.5):
            x=x1*0.5
            X.append(x)
            y=y1*0.5
            Y.append(y)
            if incircle(x, y):
                ans += 1
    return ans / n
v = []
for i in n:
    d = check(i)
    v.append(d * 4)
print(v)

plt.plot(n,v)
plt.xscale("log")
plt.figtext(.01, .99, txt , ha='left', va='top')
plt.axhline(y=3.14, color="black",linestyle="-")
plt.xlabel('Darts')
plt.ylabel('PI CAP')
plt.title('ESTIMATED PI USING MONTE CARLO')
plt.legend("PI")
plt.show()

