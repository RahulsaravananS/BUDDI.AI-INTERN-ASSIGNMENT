import numpy as np
import math
import matplotlib.pyplot as plt
a = 1
X = []
Y = []
n = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
txt="The below graph represent the monte carlo simulation to estimate the  value of pi and smoothened the curve "

def incircle(a, b):
    if (math.sqrt((a**2) + (b**2)) <= 0.5):
        return True

def check(n):
    ans = 0
    for i in range(n):
        x = np.random.uniform(-(a/2), (a/2))
        X.append(x)
        y = np.random.uniform(-(a/2), (a/2))
        Y.append(y)
        if incircle(x, y):
            ans += 1
    return ans / n

def moving_average(data, window_size):
    cumsum = np.cumsum(data)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

v = []
for i in n:
    d = check(i)
    v.append(d * 4)
print(v)
window_size = 3
smoothed_v = moving_average(v, window_size)

plt.plot(n[window_size - 1:], smoothed_v)
plt.xscale("log")
plt.figtext(.01, .99, txt , ha='left', va='top')
plt.axhline(y=3.14, color="black",linestyle="-")

plt.xlabel('Darts')
plt.ylabel('PI CAP')
plt.title('ESTIMATED PI USING MONTE CARLO')
plt.legend("PI")
plt.show()
