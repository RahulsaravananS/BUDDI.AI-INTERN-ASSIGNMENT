import numpy as np
import matplotlib.pyplot as plt
m = 0
sde=3
sdev=2
sd = 1
x = np.linspace(-5*sd, 5*sd)
f = (1 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - m) / sd) ** 2)
fev=(1 / (sdev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - m) / sdev) ** 2)
fe= (1 / (sde * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - m) / sde) ** 2)

plt.plot(x, f, color='blue', label='SD 1 Bell Curve')
plt.plot(x, fev, color='black', label='SD 2 Bell Curve')
plt.plot(x, fe, color='red', label='SD 3 Bell Curve')

plt.plot()
plt.title("Normal Distribution ")
plt.xlabel("X Value")
plt.ylabel("Probability Density")
plt.legend()
plt.figtext(0.5, 0.01, "This graph represents a standard normal distribution curve with same mean and different standard deviation.", ha="center", fontsize=10, bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})
plt.show()
