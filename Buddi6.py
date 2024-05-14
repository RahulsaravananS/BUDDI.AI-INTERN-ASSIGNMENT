# Import necessary libraries or modules
import numpy as np
import matplotlib.pyplot as plt

# Generating random y values with respect to x values
x = np.arange(-10, 10, 0.01)
# Generating y values with some noise
y = 2*x - 3 + np.random.normal(0, 5)

# Calculating polynomial coefficients using closed form
deg = 1  # Degree of polynomial
# Constructing the Vandermonde matrix for the given x values
X_mat = np.vander(x, deg + 1, increasing=True)
XT = np.transpose(X_mat)  # Transposing X_mat to facilitate matrix multiplication
XTX = np.matmul(XT, X_mat)  # X_transpose times X
XTY = np.matmul(XT, y)  # X_transpose times y
coefficients = np.matmul(np.linalg.inv(XTX), XTY)  # Closed form solution
b0_closed, b1_closed = coefficients  # Intercept and slope of the line

# Gradient Descent processing
b0_init = np.random.normal(0, 1)  # Initial guess for intercept
b1_init = np.random.normal(0, 1)  # Initial guess for slope

error_init = np.mean((y - (b0_init + b1_init*x))**2)  # Initial error calculation
lr = 0.01  # Learning rate

error = error_init  # Current error
b0 = b0_init  # Initial intercept
b1 = b1_init  # Initial slope
epoch = 0  # Initial epoch

epoch_list = [0]  # List to store epochs
error_list = [error_init]  # List to store errors

converged = False  # Flag to check convergence
B0=[]
B1=[]
# Gradient Descent loop
while not converged:
    y_pred = b0 + b1*x  # Predicted y values using current parameters
    grad_b0 = -2*np.mean((y - y_pred))  # Gradient of intercept
    grad_b1 = -2*np.mean((y - y_pred)*x)  # Gradient of slope
    # Updating parameters using gradients and learning rate
    b0 -= lr * grad_b0
    b1 -= lr * grad_b1
    # store the B0 and B1 values in list
    B0.append(b0)
    B1.append(b1)

    new_error = np.mean((y - (b0 + b1*x))**2)  # New error calculation
    epoch += 1  # Incrementing epoch
    
    # Storing epoch and error values for visualization
    epoch_list.append(epoch)
    error_list.append(new_error)
    
    # Checking convergence criteria
    if abs(error - new_error) < 10e-6:
        converged = True  # Convergence reached
    else:
        error = new_error  # Update error for next iteration

# Printing results
print("Closed Form: Beta0:", b0_closed, "Beta1:", b1_closed, "Error value:", error_init)
print("Gradient Descent: Beta0:", b0, "Beta1:", b1, "Error value:", error, "epoch:", epoch)

# Plotting error convergence

plt.plot(epoch_list, error_list,label="Error",color='blue')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Gradient Descent: Error Convergence')
plt.figtext(0.5, 0.01, "The above graph represent the error convergence of linear regression model using gradient descent", ha="center", fontsize=10, bbox={"facecolor": "brown", "alpha": 0.5, "pad": 5})
plt.legend()

#plot the 3D graph with epoch

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(B0,B1,epoch_list, cmap='viridis',label="Change of B0 and B1 with epoch",edgecolor='none')
plt.figtext(0.5, 0.01, "The above 3D graph represent the change of B0 and B1 with respect to epochs ", ha="center", fontsize=10, bbox={"facecolor": "brown", "alpha": 0.5, "pad": 5})
plt.legend()
ax.set_xlabel('Beta0(B0)')
ax.set_ylabel('Beta1(B1')
ax.set_zlabel('Epoch')
ax.set_title('Surface Plot')

# plot the 3D graph with error

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(B0,B1,error_list, cmap='viridis',label="Change of B0 and B1 with error",edgecolor='none')
plt.figtext(0.5, 0.01, "The above 3D graph represent the change of B0 and B1 with respect to error ", ha="center", fontsize=10, bbox={"facecolor": "brown", "alpha": 0.5, "pad": 5})
plt.legend()
ax.set_xlabel('Beta0(B0)')
ax.set_ylabel('Beta1(B1')
ax.set_zlabel('Error')
ax.set_title('Surface Plot')

plt.show()
