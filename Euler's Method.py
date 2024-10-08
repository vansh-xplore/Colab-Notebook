import numpy as np 
import matplotlib.pyplot as plt 

def func_slope(x, y):
    return -2 * y + (x**3) * np.exp(-2 * x)

def euler(func, x0, y0, x_end, h):
    x_values = np.arange(x0, x_end + h, h)
    y_values = [y0]

    for i in range(len(x_values) - 1):
        x = x_values[i]
        y_new = y_values[i] + h * func(x, y_values[i])
        y_values.append(y_new)
    
    return x_values, y_values

def exact_solution(x): 
    return (1/16) * (x**4) * np.exp(-2 * x) + np.exp(-2 * x)

x0 = 0
y0 = 1
x_end = 1
h = 0.1

x_values, y_values = euler(func_slope, x0, y0, x_end, h)

# Calculate the exact solution for each x in x_values
exact_y_values = exact_solution(x_values)

euler_value = y_values[-1]
exact_value = exact_solution(x_end)
error=abs(euler_value-exact_value)

print(f"Euler's Method value is: {euler_value}") 
print(f"Exact Solution value is: {exact_value}")
print(f"The error between these values is {error}")

plt.plot(x_values, y_values, label="Euler's method", color='black', marker='o', markersize=4)
plt.plot(x_values, exact_y_values, label='Exact Solution', linestyle='--', color='red')


plt.xlabel('x-axes')
plt.ylabel('y-axes')
plt.title('Comparison of Euler Method and Exact Solution')

plt.legend()
plt.grid()
plt.show() 