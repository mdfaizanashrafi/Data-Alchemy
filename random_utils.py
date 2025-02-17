#Random Number Generation : Functions for generating integers, floats, booleans, choices, and permutations.
#Statistical Distributions : Functions for uniform, normal, binomial, Poisson, and exponential distributions.
#Simulations : Functions for simulating coin tosses, dice rolls, random walks, and Monte Carlo experiments.
#Array Operations : Functions for creating arrays, performing basic operations, and slicing/indexing.
#Save and Load Data : Functions for saving and loading data to/from files.
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.bezier import inside_circle


#===========================
#RANDOM NUMBER GENERATION:
#==========================
#function to generate random integrs
def generate_random_integers(size,low,high):
    return np.random.randint(low,high,size)

#--------------------------------------------
#function to generate random floats
def generate_random_float(low=0.0,high=1.0,size=None):
    if low>=high:
        raise ValueError("Parameter 'Low' must be less than 'high'.")
    return np.random.uniform(low,high,size)

#---------------------------------------------
#function to generate random booleans
def generate_random_booleans(probability=0.5,size=None):
    if not(0.0<=probability<=1.0):
        raise ValueError("Parameter 'probability' must be in the range [0.0,1.0]")
    if size is not None:
        #Syntax: random.choice([False,True],size=size,p=[probability of false,true])
        return np.random.choice([False,True],  #possible outcomes
                                size=size, #number of samples
                                p=[1-probability,probability])  #probabilities for False and True
    #generates random numbers and then compares it to probability
    #if ran_num <prob = true else false
    return random.random() < probability

#-----------------------------------------
#function to generate random choices
def generate_random_choices(elements,size=None):
    return np.random.choice(elements,size)

#------------------------------------------
#function to generate random mutations
def generate_random_permutations(sequence):
    #parameters: sequence= list,array or range of elements to shuffle
    return np.random.permutation(sequence)

#============================
#STATISTICAL DISTRIBUTION
#=============================

#UNIFORM DISTRIBUTION
def generate_uniform_distribution(low,high,size=None):
    if low>=high:
        raise ValueError("Parameter 'low' must be less than 'high'.")
    return np.random.uniform(low,high,size)

#-------------------------------------
#NORMAL (Gaussian) DISTRIBUTION
#syntax: np.random.normal(mean,std_dev,size)
#USE: Natural Phenomena : Models traits like human heights, weights, and IQ scores.
#Measurement Errors : Represents random noise in physical measurements or sensor data.
#Statistical Analysis : Underpins hypothesis testing, confidence intervals, and regression analysis.
#Machine Learning : Simulates noise in data or initializes weights in neural networks.
#Central Limit Theorem : Approximates sums of independent random variables in large samples.

def generate_normal_distribution(loc,scale,size=None):
    if scale<=0:
        raise ValueError("Standard Deviation (std_dev) must be greater than 0")

    return np.random.normal(loc,scale,size=None)

#-------------------------------------------
#BIONOMIAL DISTRIBUTION
#no_trials=number of occurence, Prob of succ= probability
#USE:Coin Flips :Model the number of heads in n coin flips with a probability p of heads.
#Quality Control :Estimate the number of defective items in a batch of products.
#Survey Results :Predict the number of "yes" responses in a survey with n participants and a probability p of answering "yes."
#Biological Experiments :Simulate the number of successful outcomes in repeated experiments (e.g., drug trials).

def generate_binomial_distribution(n,p,size=0):
    if n<0:
        raise ValueError("Number of trials (no_trials) must be >=0.")
    if not (0<= p <=1):
        raise ValueError("Probability 'prob_of_succ' must be [0,1]")

    return np.random.binomial(n,p,size)

#------------------------------------------------------
#POISSON DISTRIBUTION
#rate: rate of occurence in specified time, size= size of returned array
#USE: Call Centers :Model the number of calls received in a given time period.
#Traffic Flow :Predict the number of vehicles passing through a point in a fixed time interval.
#Website Traffic :Estimate the number of visitors to a website in an hour.
#Rare Events :Analyze occurrences of rare events, such as equipment failures or natural disasters.

def generate_poisson_distribution(lmbda,size=None):
    if lmbda<=0:
        raise ValueError("Rate of occurance must be>=0")
    return np.random.poisson(lmbda,size)

#-----------------------------------------------------
#EXPONENTIAL DISTRIBUTION:describs time till next event
#scale-inverse of rate; size-shape of returned array
#USE: Reliability Engineering :Model the time until failure of mechanical or electronic components.
#Queueing Theory :Analyze waiting times in queues, such as customers waiting at a bank or cars at a toll booth.
#Telecommunications :Predict the time between incoming calls or data packets.
#Physics :Describe the decay of radioactive particles.

def generate_exponential_distribution(scale,size=None):
    if scale <= 0:
        raise ValueError("Scale must be more than 0")

    return np.random.exponential(scale,size)

#-----------------------------------------------------
#SIMULATE COIN TOSS
#using np.random.choice
#np.random.choice(a,size=None,replace=True,p=None)
#a:first argument is the list of options like [0,1]

def simulate_coin_toss(num_tosses,probability_heads=0.5):
    return np.random.choice([0,1],size=num_tosses,p=[1-probability_heads, probability_heads])

#---------------------------------------------------
#DICE ROLL SIMULATIONS:
#num_rolls: no. of dice roll to simulate
#num_sides: no. of sides on the dice

def simulate_dice_roll(num_rolls,num_sides=6):
    return np.random.randint(1,num_sides+1,size=num_rolls)

#-------------------------------------------------
#RANDOM WALK SIMULATION:
#the positions changes randomly eg: +1 or -1, or moving in a random direction in 2D
#for 2D: generate random steps in both X and Y axis, thn use cumsum to get cummulative pos
#num_steps=rows, 2=columns

def simulate_random_walk(num_steps, dimensions=1):
    if dimensions==1:
        steps=np.random.choice([-1,1],size=num_steps)
        return (np.cumsum(steps))

    elif dimensions==2:
        steps=np.random.choice([-1,1],size=(num_steps,2))
        return np.cumsum(steps,axis=0)
    else:
        raise ValueError("Dimension must be 1 or 2")


#MONTE CARLO SIMULATION:
#------------------------------------
#Monte Carlo Simulation is a method of using randomness to solve
#problems by simulating outcomes many times and analyzing the results.

#estimate pi value using random sampling

def estimate_pi(num_samples): #num_samples:the number of random samples to generate
    points=np.random.rand(num_samples,2) #(rows,columns)
    inside_circle= np.sum(points[:,0]**2 + points[:,1]**2 <=1)
    #equations: X^2 + Y^2 <= 1 represents inside the circle and boundary
    #Outside (0.9² + 0.9² = 0.81 + 0.81 = 1.62 > 1)
    #Inside (0.5² + 0.5² = 0.25 + 0.25 = 0.5 ≤ 1)
    return 4*inside_circle/num_samples

#=======================================================
#ARRAY OPERATIONS: Create, manipulate arrays.
#we will implement functions for creating arrays, performing basic operations
#like sum,mean,max,min), and slicing/indexing arrays
#=========================================================

#CREATE ARRAYS:

#creating zeros array
def create_zeros_array(shape):
    #parameter: shape is tuple
    return np.zeros(shape)

#Creating ones array
def create_ones_array(shape):
    return np.ones(shape)

#creating random arrays:
def create_random_array(shape):
    return np.random.rand(*shape)
    #* is used to unpack the tuple as .rand expects seperate argument for each dimension

#PERFORM BASIC OPERATIONS:
#Calculate Sum
def calculate_sum(array):
    return np.sum(array)

#calculate mean
def calculate_mean(array):
    return np.mean(array)

#find maximum
def find_max(array):
    return np.max(array)

#find minimum
def find_min(array):
    return np.min(array)

#=======================================================
#SLICE AND INDEX ARRAYS:
#================================================
#get first element:
def get_first_element(array):
    return array[0]

#get last element:
def get_last_element(array):
    return array[-1]

#SLICING ARRAY
def slice_array(array,start,end):
    return array[start:end]

#===================================================
#VISUALIZATION:
#=================================================

#ARRAY VISUALIZATION: plot arrays:
def plot_array(array,title="Array Visualization"):
    plt.plot(array,marker='o')
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()

#VISUALIZE RANDOM WALK SIMULATION:
#for 1D walks, it plots the position over time.
#2D walks, it plots the path in a 2D plane.

def visualize_random_walk(positions, title= "Random Walk Simulation"):
    if positions.ndim ==1:
        plt.plot(positions,marker='o')
        plt.title(title)
        plt.xlabel("Step")
        plt.ylabel("Positions")
        plt.grid(True)
    elif positions.ndim == 2:
        plt.plot(positions[:,0],positions[:,1],marker='o')
        plt.title(title)
        plt.xlabel("X-Position")
        plt.ylabel("Y-Position")
        plt.grid(True)
    else:
        raise ValueError("Positions must be 1D or 2D.")
    plt.show()

#CUSTOMIZE VISUALIZATION
#Histograms: visualize the distribution of random numbers
#Scatter Plots: Plot relationships between two variables
#Bar Charts: Display Categorical Data















