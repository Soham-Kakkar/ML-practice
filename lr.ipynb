import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Perform linear regression on a CSV file.')
parser.add_argument('filename', type=str, help='The name of the CSV file (with or without .csv extension)')
parser.add_argument('learning_rate', type=float, help='The learning rate for the regression')
parser.add_argument('epochs', type=int, help='The number of epocks for the regression')

# Parse the arguments
args = parser.parse_args()
filename = args.filename.lower()
lr = args.learning_rate
iterations = args.epochs

# Load the data
try:
    data = pd.read_csv(filename if filename.endswith('.csv') else filename + '.csv')
except FileNotFoundError:
    print("File not found. Please ensure the file is in the same directory as the script.")
    exit()
except pd.errors.EmptyDataError:
    print("File is empty. Please ensure the file contains data.")
    exit()
except pd.errors.ParserError:
    print("Error parsing the file. Please ensure the file is a valid CSV.")
    exit()

# Ensure the data types are float for calculations
data[data.columns[0]] = data[data.columns[0]].astype(float)
data[data.columns[1]] = data[data.columns[1]].astype(float)

# Get column names for labeling
x_label = data.columns[0]  # First column as x
y_label = data.columns[1]  # Second column as y

# Feature Scaling (Min-Max Normalization)
x_min = data[x_label].min()  # First column as x
x_max = data[x_label].max()
y_min = data[y_label].min()  # Second column as y
y_max = data[y_label].max()

data[x_label] = (data[x_label] - x_min) / (x_max - x_min)  # Normalize x
data[y_label] = (data[y_label] - y_min) / (y_max - y_min)  # Normalize y

# Gradient Descent function
def grad_descent(m, b, points, l):
    grad_m = 0
    grad_b = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i][x_label]  # Access x
        y = points.iloc[i][y_label]  # Access y
        grad_m += (2 / n) * (y - (m * x + b)) * (-x)
        grad_b += (2 / n) * (y - (m * x + b)) * (-1)

    m_new = m - l * grad_m
    b_new = b - l * grad_b
    return m_new, b_new  # Return the updated values of m and b

# Initialize parameters
m = 0
b = 0
l = lr  # Learning rate
epochs = iterations  # Number of epochs

# Perform Gradient Descent
for i in range(epochs):
    m, b = grad_descent(m, b, data, l)

# Print final values of m and b in normalized space
print(f"Final normalized m: {m}, Final normalized b: {b}")

# Reverse normalization for slope (m) and intercept (b)
m_original = m * (y_max - y_min) / (x_max - x_min)
b_original = b * (y_max - y_min) + y_min - m_original * x_min

# Print final values of m and b in original space
print(f"Final original m: {m_original}, Final original b: {b_original}")

# Plot the regression line
plt.scatter(data[x_label] * (x_max - x_min) + x_min, 
            data[y_label] * (y_max - y_min) + y_min, 
            label='Data Points')

# Create x values for the regression line in original scale
x_vals = np.linspace(x_min, x_max, len(data[x_label]))
y_vals = m_original * x_vals + b_original  # Adjust for plotting

plt.plot(x_vals, y_vals, color='red', label='Regression Line')
plt.xlabel(x_label)  # Use dynamic x label
plt.ylabel(y_label)  # Use dynamic y label
plt.title(f'{x_label} vs {y_label} with Regression Line')
plt.legend()
plt.grid()
plt.show()
