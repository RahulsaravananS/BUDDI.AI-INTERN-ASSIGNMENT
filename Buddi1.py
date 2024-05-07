import numpy as np
import matplotlib.pyplot as plt

X = np.array([-3, -2, -1, 0, 1, 2, 3])
Y = np.array([7, 2, 0, 0, 0, 2, 7])

beta1= []
beta2 = []
E = []

MinEpsilon = 2 ** 30
B1 = -1
B2 = -1

for i in range(-3,4,1):
   i=i
   for j in range(-3,4,1):
      j=j
      b1 = i
      b2 = j
      count = 0
      beta1.append(i)
      beta2.append(j)
      for k in range(len(X)):
          Val = (b1 * X[k]) + (b2 * (X[k]) ** 2)
          count += abs(Y[k] - Val)
      E.append(count)
      if count < MinEpsilon:
          MinEpsilon = count
          B1 = b1
          B2 = b2  

print(f"MinEpsilon is {MinEpsilon} and it occurs when B1 is {B1} and B2 is {B2}")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(beta1, beta2,E, cmap='viridis', edgecolor='none')

ax.set_xlabel('Beta1(B1)')
ax.set_ylabel('Beta2(B2)')
ax.set_zlabel('Epsilon(E)')
ax.set_title('Surface Plot')

plt.show()
