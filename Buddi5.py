# Import necessary libraries or modules
import numpy as np
import matplotlib.pyplot as plt

# Define the Lagrange Interpolation function
def lagrangeInterpolation(x, y, xInterp):
    # Number of known points
    n = len(x)
    # Number of points to interpolate
    m = len(xInterp)
    # Initialize array to store interpolated y values
    yInterp = np.zeros(m)
    
    # Loop over each point to interpolate
    for j in range(m):
        p = 0  # Initialize the interpolated value
        # Loop over each known point
        for i in range(n):
            L = 1  # Initialize Lagrange polynomial for this known point
            # Calculate Lagrange polynomial for this known point
            for k in range(n):
                if k != i:
                    L *= (xInterp[j] - x[k]) / (x[i] - x[k])
            # Add the contribution of this known point to the interpolated value
            p += y[i] * L
        # Store the interpolated value for this point
        yInterp[j] = p
    # Return the array of interpolated y values
    return yInterp

# Known points
x = np.array([-2, -1, 0, 1, 2])
y = np.array([4, 1, 0, 1, 4])

# Points to interpolate
xInterp = np.linspace(-2, 2, 100)

# Perform Lagrange interpolation
yInterp = lagrangeInterpolation(x, y, xInterp)

# Plot the known points and interpolated curve
plt.plot(x, y, 'bo', label='Known Points')  # Known points
plt.plot(xInterp, yInterp, 'r-', label='Lagrange Interpolation')  # Interpolated curve
plt.xlabel('X value')
plt.ylabel('Y value')
plt.title('Lagrange Interpolation Polynomial')
plt.legend()
# Add a text box with additional information
plt.figtext(0.5, 0.01, "This graph represents the Lagrange interpolation polynomial which covers all the known data points.", ha="center", fontsize=10, bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})
plt.show()
