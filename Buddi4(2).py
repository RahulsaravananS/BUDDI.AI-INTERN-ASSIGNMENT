import numpy as np
import matplotlib.pyplot as plt
m = 0
sde=3
sdev=2
sd = 1
x1 = np.linspace(m-(5*sd), m+(5*sd))
x2 = np.linspace(m-(5*sdev), m+(5*sdev))
x3 = np.linspace(m-(5*sde), m+(5*sde))

f = (1 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x1 - m) / sd) ** 2)
fev=(1 / (sdev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x2 - m) / sdev) ** 2)
fe= (1 / (sde * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x3 - m) / sde) ** 2)

plt.plot(x1, f, color='blue', label='SD 1 Bell Curve')
plt.plot(x2, fev, color='black', label='SD 2 Bell Curve')
plt.plot(x3, fe, color='red', label='SD 3 Bell Curve')

plt.plot()
plt.title("Normal Distribution ")
plt.xlabel("X Value")
plt.ylabel("Probability Density")
plt.legend()
plt.figtext(0.5, 0.01, "This graph represents a standard normal distribution curve with same mean and different standard deviation.", ha="center", fontsize=10, bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})
plt.show()
