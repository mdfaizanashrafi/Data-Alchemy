#Random Number Generation : Functions for generating integers, floats, booleans, choices, and permutations.
#Statistical Distributions : Functions for uniform, normal, binomial, Poisson, and exponential distributions.
#Simulations : Functions for simulating coin tosses, dice rolls, random walks, and Monte Carlo experiments.
#Array Operations : Functions for creating arrays, performing basic operations, and slicing/indexing.
#Save and Load Data : Functions for saving and loading data to/from files.
import random

import numpy as np

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






