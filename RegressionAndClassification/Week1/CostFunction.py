# Optional lab: Cost function
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

print(np.__version__)

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
w = 200
b = 100

def compute_cost(x, y, w, b):
    m = x.shape[0]

    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost
    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost

cost = compute_cost(x_train, y_train, w, b)
print(cost)

# Create a arry for Plot our cost function
wArray = np.arange(0, 400)

def compute_model_output(x, y, w, b):
    m = w.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = compute_cost(x, y, i, b)
    return f_wb

# Plot our cost function
tmp_f_cost = compute_model_output(x_train, y_train, wArray, b)
plt.plot(wArray, tmp_f_cost, c='b',label='Plot our cost function')
# Set the title
plt.title("Plot our cost function")
# Set the y-axis label
plt.ylabel('Cost')
# Set the x-axis label
plt.xlabel('w')
plt.show()