import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def load_mnist_data(file_path):
    """
    Load MNIST data from a CSV file.

    Parameters: 
    file_path (str): The path to the CSV file containing MNIST data.    
    dffsdff
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Separate features and labels
    X = df.iloc[:, 1:].values  # Features (pixel values)
    y = df.iloc[:, 0].values   # Labels (digits)
    print(df.head(10).to_string())
    return X, y

X, y = load_mnist_data('./mnist_test.csv')

print(X.shape)
print(y)

# 1. Create a 28x28 NumPy array of random zeros and ones
np.random.seed(43) # for a reproducible, random image
data = np.random.randint(0, 2, size=(28, 28))

# 2. Use matplotlib.pyplot.imshow to display this array
plt.figure(figsize=(5, 5))
# cmap='binary' ensures 0 is black and 1 is white, displaying the grid
plt.imshow(data, cmap='binary')

# 3. Remove the axes for a clean, grid-like appearance
plt.axis('off')
