import numpy as np
import matplotlib.pyplot as plt
from fontTools.varLib.instancer import axisValuesFromAxisLimits

from random_utils import simulate_random_walk, estimate_pi, create_zeros_array, create_ones_array, create_random_array, \
    calculate_sum, calculate_mean, find_max, find_min, get_first_element, get_last_element, slice_array, plot_array, \
    visualize_random_walk, plot_histogram, save_data_to_file, reshape_array, concatenate_array, split_array, \
    transpose_array, apply_trigonometric, apply_exponential, apply_custom_operation, matrix_multiply, \
    solve_linear_equation, compute_eigen, compute_determinant, compute_inverse, calculate_variance, calculate_median, \
    compute_correlation, compute_covariance, demonstrate_broadcasting, standardize_array, plot_heatmap, plot_3d
from random_utils import simulate_coin_toss, simulate_dice_roll



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

#Create a random 1D array and reshape it into a 2D array:
random_array_for_reshape = create_random_array((9,))
print(f"Array to Reshape: {random_array_for_reshape}")

reshaped_array = reshape_array(random_array_for_reshape,(3,3))
print(f"Reshaped Array: {reshaped_array}")

#Create two random array and concatenate them:
array1 = create_random_array((3,))
array2 = create_random_array((3,))
print(f"Array 1: {array1}")
print(f"Array 2: {array2}")

#calling the concatenation function from random-utils
concatenated_array = concatenate_array(array1,array2)
print(f"Concatenated Array: {concatenated_array}")

#create a random array and split it into 3 parts:
random_array_split = create_random_array((9,))
print(f"Original array: {random_array_split}")

splitted_arrays = split_array(random_array_split,num_splits=3)
print("Split Arrays:")
for i, part in  enumerate(splitted_arrays):
    print(f"Part {i+1}: {part}")

#Create a 2D array and transpose it
random_array_transpose= create_random_array((3,3))
print(f"Original Array to Transpose: {random_array_transpose}")

transposed_array = transpose_array(random_array_transpose)
print(f"Transposed Array: {transposed_array}")

#=======================
#SLICING
#=======================

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

#================================
#TRIGONOMETRIC FUNCTIONS:
#================================
#create a random array and apply sine,cose,tan function
random_trigo_array = create_random_array((5,))
print(f"Original Trigo Array: {random_trigo_array}")

sin_result = apply_trigonometric(random_trigo_array,func='sin')
cos_result = apply_trigonometric(random_trigo_array,func='cos')
tan_result = apply_trigonometric(random_trigo_array,func='tan')

print(f"Sin Result: {sin_result}")
print(f"Cos Result: {cos_result}")
print(f"Tan Result: {tan_result}")

#EXPONENTIAL AND LOGARITHMIC FUNCTIONS:
#create a random array and apply exponential,nat log, log @ base x
array_for_log = create_random_array((5,))
print(f"Original Array for Log operation: {array_for_log}")

exponential_result = apply_exponential(array_for_log,operation='exp')
log_result = apply_exponential(array_for_log,operation='log')
log10_result = apply_exponential(array_for_log, operation='log10')
log_base_result = apply_exponential(array_for_log,operation='log_base',base=3)

print(f"Exponential Log: {exponential_result}")
print(f"Natural Log: {log_result}")
print(f"Log at Base 10: {log10_result}")
print(f"Log at Base x: {log_base_result}")

#create a random array and apply a custom operation
random_array_for_custom_operartion = create_random_array((5,))
print(f"Original Array for Custom Operation: {random_array_for_custom_operartion}")

squared_result = apply_custom_operation(random_array_for_custom_operartion,operation=lambda x:x**2)
sqrt_result = apply_custom_operation(random_array_for_custom_operartion, operation=lambda x: np.sqrt(x))

print(f"Squared Operation: {squared_result}")
print(f"Square Root Operation: {sqrt_result}")

#create two matrix for multiplication
matrix1 = np.array([[1,2],[3,4]])
matrix2 = np.array([[3,4],[5,6]])

print(f"Matrix 1: \n {matrix1}")
print(f"Matrix 2: \n {matrix2}")

multipled_matrix = matrix_multiply(matrix1, matrix2)
print(f"Matrix Multiplied: \n {multipled_matrix}")

#Define the coefficent matrix and constants vector
A=np.array([[3,1],[1,2]])
b=np.array([9,8])

print(f"Coeffiecient Matrix (A): \n {A}")
print(f"Constant Vector (b):\n {b}")

solution_to_linear_eq= solve_linear_equation(A,b)
print(f"Solution Vector for LinEq: \n {solution_to_linear_eq}")

#create a square matrix and compute eigenvalues and eigenvectors
matrix_eigen = np.array([[4,2],[1,3]])
print(f"Matrix: \n {matrix_eigen}")

eigenvalues, eigenvectors = compute_eigen(matrix_eigen)
print(f"Eigevalues: {eigenvalues}")
print(f"Eigenvectors: {eigenvectors}")

#DETERMINANTS AND INVERSE:

#create square matrix
matrix_dete_inv = np.array([[4,7],[2,6]])
print(f"Matrix:\n {matrix_dete_inv}")

determinant = compute_determinant(matrix_dete_inv)
inverse = compute_inverse(matrix_dete_inv)

print(f"Determinant:\n {determinant}")
print(f"Inverse:\n {inverse}")

#===========================================
#STATISTICS: 1) Median and Variance
#            2) Correlation and Covariance
#============================================
#Median and Variance
#create a random array
median_variance_array = create_random_array((10,))
print(f"Random Array: {median_variance_array}")

median = calculate_median(median_variance_array)
variance = calculate_variance(median_variance_array)

print(f"Median: {median}")
print(f"Variance: {variance}")

#CORRELATION AND COVARIANCE

#create two random arrays:
corr_cov_array1 = create_random_array((5,))
corr_cov_array2 = create_random_array((5,))

print(f"Array 1: {corr_cov_array1}")
print(f"Array 2: {corr_cov_array2}")

correlation = compute_correlation(corr_cov_array1,corr_cov_array2)
covariance = compute_covariance(corr_cov_array1,corr_cov_array2)

print(f"Correlation Coefficeint: {correlation}")
print(f"Covariance: {covariance}")

#BROADCASTING:

#broadcast a 1D array across a 2D array
array_2d = np.array([[1,2,3],[4,5,6]])
array_1d = np.array([10,20,30])

print(f"2D Array:\n {array_2d}")
print(f"1D Array: \n {array_1d}")

result = demonstrate_broadcasting(array_2d,array_1d)
print(f"Result (2D Array + 1D Array):\n {result}")

#================================================
#CUSTOM UTILITIES:
#=================================================
#Normalize:
#create a random array to normalize it
normalize_random_array = create_random_array((5,))
print(f"Original Array to Normalize: {normalize_random_array}")

from random_utils import normalize_array
normalized_array = normalize_array(normalize_random_array)
print(f"Array after Normalization: {normalized_array}")

#Scaling:
#create a random array and standardize it
scaling_random_array = create_random_array((5,))
print(f"Original array to scale/standardize: {scaling_random_array}")

standardized_array= standardize_array(scaling_random_array)
print(f"Standardized Array: {standardized_array}")

#========================================
#ADVANCED VISUALIZATION:
#========================================
#HEATMAP
# create a random 2D array and plot it as a heatmap
heatmap_plot = np.random.rand(5,5)
plot_heatmap(heatmap_plot,title="Random Matrix Heatmap")

# 3D plotting
#create a 3D surface plot of Z = sin(sqrt(x^2 + y^2))

x= np.linspace(-5,5,100)
y = np.linspace(-5,5,100)
X, Y = np.meshgrid(x,y)
Z = np.sin(np.sqrt(X**2 + Y**2))

plot_3d(X,Y,Z,title="3D Surface Plot")













