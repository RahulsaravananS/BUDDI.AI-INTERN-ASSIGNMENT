import numpy as np
import matplotlib.pyplot as plt
m = 0
me=1
mea=2
sd = 1
x1 = np.linspace(m-(5*sd), m+(5*sd))
x2 = np.linspace(me-(5*sd), me+(5*sd))
x3 = np.linspace(mea-(5*sd), mea+(5*sd))

f = (1 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x1 - m) / sd) ** 2)
fe= (1 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x2 - me) / sd) ** 2)
fea = (1 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x3 - mea) / sd) ** 2)

plt.plot(x1, f, color='blue', label='mean 0 Bell Curve')
plt.plot(x2, fe, color='red', label='mean 1 Bell Curve')
plt.plot(x3, fea, color='black', label='mean 2 Bell Curve')

plt.plot()
plt.title("Normal Distribution ")
plt.xlabel("X Value")
plt.ylabel("Probability Density")
plt.legend()
plt.figtext(0.5, 0.01, "This graph represents a standard normal distribution curve with different mean and same standard deviation.", ha="center", fontsize=10, bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})
plt.show()
