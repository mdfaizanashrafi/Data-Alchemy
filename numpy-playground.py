import numpy as np
import matplotlib.pyplot as plt

from random_utils import (
    generate_random_choices,
    generate_random_booleans,
    generate_random_permutations,
    generate_random_integers,
    generate_random_float,
    generate_uniform_distribution,
    generate_normal_distribution,
    generate_binomial_distribution,
    generate_poisson_distribution,
    generate_exponential_distribution

)










'''
#Test Random Choices
print(f"Random Distribution: {(generate_random_choices(4,2))}")

#Test Random Booleans
print(f"Random Booleans: {generate_random_booleans()}")

#Test Random Permutations
print(f"Random Permutations: {(generate_random_permutations([1,4,3,5,7,4,2]))}")

#Test Random Integers
print(f"Random Intergers: {(generate_random_integers((2,3),3,7))}")

#Test Random Float
print(f"Random Float: {generate_random_float(1,2,5)}")

# Test Uniform Distribution
print(f"Uniform Distribution: {generate_uniform_distribution(low=0, high=10, size=5)}")

# Test Normal Distribution
print("Normal Distribution:", generate_normal_distribution(loc=0, scale=1, size=5))

# Test Binomial Distribution
print("Binomial Distribution:", generate_binomial_distribution(n=10, p=0.5, size=5))

# Test Poisson Distribution
print("Poisson Distribution:", generate_poisson_distribution(lmbda=3, size=5))

# Test Exponential Distribution
print("Exponential Distribution:", generate_exponential_distribution(scale=2, size=5))
======================================================================
def create_array():
    print("Welcome to the NumPy Playground!")

    # Step 1: Create an array
    size = int(input("Enter the size of the array: "))
    arr = np.random.randint(0, 100, size)  # Random integers between 0 and 100
    print(f"Your array: {arr}")

    # Step 2: Perform basic operations
    print("\nWhat would you like to do?")
    print("1. Find the mean")
    print("2. Find the maximum value")
    print("3. Find the minimum value")
    choice = input("Enter your choice (1/2/3): ")

    if choice == "1":
        result = np.mean(arr)
        operation = "mean"
    elif choice == "2":
        result = np.max(arr)
        operation = "maximum"
    elif choice == "3":
        result = np.min(arr)
        operation = "minimum"
    else:
        result = "Invalid choice"
        operation = None

    if operation:
        print(f"The {operation} of the array is: {result}")

    # Step 3: Visualize the array
    print("\nVisualizing the array...")
    plt.plot(arr, marker='o')
    plt.title("Array Visualization")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()


# Run the function
create_array()

'''