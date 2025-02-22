import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

# Load the CSV data (make sure your CSV file has columns 'x' and 'y')
data = pd.read_csv('4x4_prices.csv')

# Extract the x and y values
x = data['4x4 length']
y = data['price']

# Create a scatter plot of the data
plt.scatter(x, y, color='black')

# Fit a linear model using a first degree polynomial (y = mx + b)
coeffs = np.polyfit(x, y, 1)
slope, intercept = coeffs

# Print the equation of the linear model
print(f"Linear model: y = {slope:.3f}x + {intercept:.3f}")

# Generate y-values based on the fitted model for plotting
y_fit = slope * x + intercept

# Plot the linear fit line
plt.plot(x, y_fit, color='blue', label='Price per Foot')

# Add labels and legend to the plot
plt.xlabel('4x4 Length')
plt.ylabel('Price ($)')
plt.title('4x4 Price vs Length')
plt.legend()

# Display the plot
plt.show()
