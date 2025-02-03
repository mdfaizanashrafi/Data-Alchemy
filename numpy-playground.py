import numpy as np
import matplotlib.pyplot as plt


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