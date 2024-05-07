import numpy as np
import matplotlib.pyplot as plt
m = 0
me=1
mea=2
sd = 1
x = np.linspace(-5*sd, 5*sd)
f = (1 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - m) / sd) ** 2)
fe= (1 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - me) / sd) ** 2)
fea = (1 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mea) / sd) ** 2)

plt.plot(x, f, color='blue', label='mean 0 Bell Curve')
plt.plot(x, fe, color='red', label='mean 1 Bell Curve')
plt.plot(x, fea, color='black', label='mean 2 Bell Curve')

plt.plot()
plt.title("Normal Distribution ")
plt.xlabel("X Value")
plt.ylabel("Probability Density")
plt.legend()
plt.figtext(0.5, 0.01, "This graph represents a standard normal distribution curve with different mean and same standard deviation.", ha="center", fontsize=10, bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})
plt.show()
