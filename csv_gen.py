import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)

# Function to generate data based on the specified model
def generate_data(model='linear', num_points=100):
    x = np.random.uniform(0, 10, num_points)
    
    noise = np.random.normal(0, np.random.uniform(0.5, 2), num_points)

    if model == 'linear':
        y = 2 * x + noise
    elif model == 'quadratic':
        y = 0.5 * x**2 + noise
    elif model == 'quartic':
        y = 0.1 * x**4 - 0.5 * x**3 + noise
    elif model == 'sinusoidal':
        y = np.sin(x) * np.random.uniform(0.5, 1.5) + noise
    elif model == 'exponential':
        y = np.exp(0.1 * x) + noise
    elif model == 'logarithmic':
        y = np.log(x + 1) + noise  # Adding 1 to avoid log(0)
    else:
        raise ValueError("Unsupported model type. Choose from 'linear', 'quadratic', 'quartic', 'sinusoidal', 'exponential', or 'logarithmic'.")
    
    return x, y

# Change the model type here
model_type = input("What variation type do you expect: ").lower()  # Change this to 'quadratic', 'quartic', or 'sinusoidal' as needed
x, y = generate_data(model=model_type)

df = pd.DataFrame({'x': x, 'y': y})

filename = input("filename: ").strip().lower()
df.to_csv(filename if filename.endswith('.csv') else filename + '.csv', index=False)

plt.scatter(x, y)
plt.title(f'{model_type.capitalize()} Data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
