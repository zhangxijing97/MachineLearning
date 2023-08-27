#Gradient descent
# Implement Gradient Descent:
# 1. compute_cost
# 2. compute_gradient
# 3. gradient_descent

import math, copy
import numpy as np
print(np.__version__)
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

# Load our data set
x_train = np.array([1.0, 2.0])   #features
y_train = np.array([300.0, 500.0])   #target value

# 1. compute_cost
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i]) ** 2
    total_cost = 1 / (2 * m) * cost

    return total_cost

# 2. compute_gradient
# Conventions:
# 1. ∂𝐽(𝑤,𝑏)/∂𝑏 is dj_db
# 2. w.r.t is With Respect To
def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression
    Args:
      x (ndarray (m,)): Data, m examples
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b
     """

    # Number of training examples
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i] # ∂𝐽(𝑤,𝑏)/∂w = 1/m*∑( fw,b(x(i)) - y(i) )* x(i)
        dj_db_i = f_wb - y[i] # ∂𝐽(𝑤,𝑏)/∂𝑏 = 1/m*∑( fw,b(x(i)) - y(i) )
        dj_db += dj_db_i # update derivative of ∂𝐽(𝑤,𝑏)/∂b
        dj_dw += dj_dw_i # update derivative of ∂𝐽(𝑤,𝑏)/∂w
    dj_dw = dj_dw / m # complete last step: 1/m
    dj_db = dj_db / m # complete last step: 1/m

    return dj_dw, dj_db
    # The gradient of the cost with respect to the parameters w or b
    # gradient = derivative = rate of change of a function

# 3. gradient_descent
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking
    num_iters gradient steps with learning rate alpha

    Args:
      x (ndarray (m,))  : Data, m examples
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient

    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b]
      """

    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w, b) # get ∂𝐽(𝑤,𝑏)/∂w and ∂𝐽(𝑤,𝑏)/∂b

        # Update Parameters using equation (3) above
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            J_history.append(cost_function(x, y, w, b)) # store all total_cost
            p_history.append([w, b]) # store all w,b
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, J_history, p_history  # return w and J,w history for graphing

# initialize parameters
w_init = 0
b_init = 0
# some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2 # learning rate
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")