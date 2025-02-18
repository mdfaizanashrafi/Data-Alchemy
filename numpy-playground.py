import numpy as np
import matplotlib.pyplot as plt
from random_utils import simulate_random_walk, estimate_pi, create_zeros_array, create_ones_array, create_random_array, \
    calculate_sum, calculate_mean, find_max, find_min, get_first_element, get_last_element, slice_array, plot_array, \
    visualize_random_walk, plot_histogram, save_data_to_file
import random_utils



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
#======================================================================
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

#====================================================
#SIMULATIONS:
#====================================================

#Simulate coin toss
outcomes_coins= simulate_coin_toss(num_tosses=10,probability_heads=0.7)
print(f"Coin Toss Outcomes: {outcomes_coins}")

#simulate dice roll
outcomes_dice=simulate_dice_roll(num_rolls=5)
print(f"Dice Roll Outcomes: {outcomes_dice}")


#simulate random walk
#simulate a 1D random walk with 10 steps:
positions_1d = simulate_random_walk(num_steps=10,dimensions=1)
print(f"1D random walk positons: {positions_1d}")

#simulate a 2D random walk with 10 steps:
positions_2d= simulate_random_walk(num_steps=10,dimensions=2)
print(f"2D random walk positions:\n {positions_2d}")

#Estimate Value of pi using 1,000,000 random samples
pi_estimate = estimate_pi(num_samples=1000000)
print(f"Estimated Value of pi: {pi_estimate}") #Output: Estimated Value of pi: 3.140456

#create a 3x3 array of zeros
zero_array=create_zeros_array((3,3))
print(f"Zeros array:\n {zero_array}")

#create a 3x3 array of ones
ones_array= create_ones_array((3,3))
print(f"Ones array:\n {ones_array}")

random_array_create=create_random_array((3,3))
print(f"Random array:\n {random_array_create}")

#crate random arrays and calculate its sum
random_array_sum= create_random_array((3,3))
total_sum= calculate_sum(random_array_sum)
print(f"Random array:\n {random_array_sum}")
print(f"Sum of elements: \n {total_sum}")

#create random array and calculate its mean
random_array_mean=create_random_array((3,3))
mean_value=calculate_mean(random_array_mean)
print(f"Mean of the random array:\n {mean_value}")

#create random array and calculate its max
random_array_max = create_random_array((3,3))
max_value= find_max(random_array_max)
print(f"Random arrays:\n {random_array_max}")
print(f"Max value in the random array:\n {max_value}")

#create a random array and find its minimum value
random_array_min = create_random_array((3,3))
min_value= find_min(random_array_min)
print(f"Minimum value in the array:\n {min_value}")

#get first and last element of a generated array
random_array_for_first_and_last_element= create_random_array((3,3))
first_element= get_first_element(random_array_for_first_and_last_element)
last_element= get_last_element(random_array_for_first_and_last_element)
print(f"Random array:\n {random_array_for_first_and_last_element}")
print(f"First element:\n {first_element}")
print(f"Last element:\n {last_element}")

#SLICING
#create random array to slice it
random_array_slice = create_random_array((5,))
sliced_array= slice_array(random_array_slice,1,3)
print(random_array_slice) #output:[0.10214748 0.00398919 0.84170935 0.72114834 0.78040356]
print(sliced_array) #output:[0.28630149 0.32521568]

#MATPLOTLIBRARY:

random_array_plot= create_random_array((10,))
plot_array(random_array_plot,title="Random Array Visualization")

#VISUALIZE RANDOM WALK
#simulate and visualize 1D random walk

position_1d = simulate_random_walk(num_steps=20,dimensions=1)
visualize_random_walk(position_1d,title="1-D random walk")

#simulate and visualize 2D random walk

position_2d= simulate_random_walk(num_steps=20, dimensions=2)
visualize_random_walk(position_2d,title="2D Random Walk Simulation")

#create a random array and plot its histogram
random_array_hist= create_random_array((1000,))
plot_histogram(random_array_hist,title="Random Data Histogram")

#COMBINED VISUALIZATION

#generate and visualize a random array
random_arary= create_random_array((10,))
print(f"Mean of Array: {calculate_mean(random_arary)}")
plot_array(random_arary,title="Random Array Visualization")

#Simulate and visualize a 2D random walk
positions_2d= simulate_random_walk(num_steps=50,dimensions=2)
visualize_random_walk(positions_2d,title="2D Random Walk")

#create a random array and save it to a file
random_array = create_random_array((5,))
save_data_to_file(random_array,"data/random_array.txt",format="txt")
save_data_to_file(random_array, "data/random_array.csv",format="csv")
print("Data saved to files")





