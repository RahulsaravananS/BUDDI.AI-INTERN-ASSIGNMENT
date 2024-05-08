import numpy as np
import matplotlib.pyplot as plt

def lagrangeInterpolation(x, y, xInterp):
    n = len(x)
    m = len(xInterp)
    yInterp = np.zeros(m)
    
    for j in range(m):
        p = 0
        for i in range(n):
            L = 1
            for k in range(n):
                if k != i:
                    L *= (xInterp[j] - x[k]) / (x[i] - x[k])
            p += y[i] * L
        yInterp[j] = p
    return yInterp

x = np.array([-2, -1, 0, 1, 2])
y = np.array([4, 1, 0, 1, 4])
xInterp = np.linspace(-2, 2, 100)
yInterp = lagrangeInterpolation(x, y, xInterp)

plt.plot(x, y, 'bo', label='Known Points')
plt.plot(xInterp, yInterp, 'r-', label='Lagrange Interpolation')
plt.xlabel('X value')
plt.ylabel('Y value')
plt.title('Lagrange Interpolation Polynomial')
plt.legend()
plt.figtext(0.5, 0.01, "This graph represent the lagrange interpolation polynomial which covers all the known data points.", ha="center", fontsize=10, bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})
plt.show()