#! /usr/bin/env python
import pooltool as pt
import numpy as np
from pooltool.ruleset.three_cushion import is_point
from pooltool.events.datatypes import Event, EventType
from pooltool.events.filter import by_ball, by_time, by_type, filter_events, filter_ball, filter_type
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
import time

import random
from deap import base, creator, tools, algorithms
from functools import partial
from threecushion_shot import BilliardEnv

shotnums = 20

sidespin_stddev = 0.025
vertspin_stddev = 0.025
cuespeed_stddev = 0.04
cutangle_stddev = 2
cueincline_stddev = 1

var_stddev = [sidespin_stddev, vertspin_stddev, cuespeed_stddev, cutangle_stddev, cueincline_stddev]

# Step 1: Define the optimization problem
# - Maximize the result (creator, fitness)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


# Step 2: Define the individual structure (7 variables in this case)
initial_individuals = [0, 0, 0, 0, 0]  # Initial guess for the variables

# Step 3: Define the objective function (maximize)
def objective_function(individual, shotnums, var_stddev):
    # Define your objective function here
    # Example: A simple nonlinear function like sum of squares
    # individual stores every input variable in a vector

    env = BilliardEnv(shotnums, var_stddev)

    successrate = env.shot_randomize(individual)

    return successrate,

def create_individual():
    return [random.uniform(-1, 1) for _ in range(5)]

# Step 4: Set up genetic algorithm (population, selection, crossover, mutation)
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", partial(objective_function, shotnums=shotnums, var_stddev=var_stddev))
toolbox.register("mate", tools.cxBlend, alpha=0.1)  # Crossover: blend two individuals
toolbox.register("select", tools.selTournament, tournsize=3)  # Selection: Tournament selection

# Step 5: Initialize the population
population = toolbox.population(n=1000)  # Create a population of 100 individuals

# Step 6: Define the algorithm parameters
generations = 50  # Number of generations to run
cxpb = 0.7  # Crossover probability
mutpb = 0.2  # Mutation probability

# Step 7: Run the algorithm
for gen in range(generations):
    if gen < generations // 2:
        sigma = 1.0  # High mutation rate
    else:
        sigma = 0.2  # Low mutation rate
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=sigma, indpb=0.2)  # Mutation: Gaussian mutation

    # Evaluate fitness of the population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Select the next generation
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cxpb:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < mutpb:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the fitness of individuals with invalid fitness values
    invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid_individuals))
    for ind, fit in zip(invalid_individuals, fitnesses):
        ind.fitness.values = fit

    # Replace the old population with the new one
    population[:] = offspring

    # Print the best solution in the current generation
    top_individual = tools.selBest(population, 1)[0]
    # denormalize the variables
    sidespin_avg, vertspin_avg, cuespeed_avg, cutangle_avg, cueincline_avg = self.denormalize(top_individual)

    print(f"Generation {gen + 1}, Best Solution: {top_individual}, Fitness: {top_individual.fitness.values[0]}")
    print(f"ss=", np.round(sidespin_avg, 3),
          ", vs=", np.round(vertspin_avg, 3),
          ", speed=", np.round(cuespeed_avg, 3),
          ", cut=", np.round(cutangle_avg, 3),
          ", incline=", np.round(cueincline_avg, 3))
