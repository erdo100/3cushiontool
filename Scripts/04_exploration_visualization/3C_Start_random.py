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

# Example usage
num = 100000  # Length of each random variable list
limits = [
    (-0.5, 0.5),   # a
    (-0.5, 0.5),  # b
    (2, 7), # velocity
    (-89, 90),   # cut angle
    (0, 20) # incline angle
]

# Define all variables
random_variables = []
lower_limit, upper_limit = limits[i]
a_rand = np.random.uniform(limits[0][0], limits[0][1], num)
b_rand = np.random.uniform(limits[1][0], limits[1][1], num)
v_rand = np.random.uniform(limits[2][0], limits[2][1], num)
cut_rand = np.random.uniform(limits[3][0], limits[3][1], num)
inc_rand = np.zeros(num)

env = BilliardEnv(shotnums, var_stddev)

successrate = env.shot_randomize(individual)


print(f"Generation {gen + 1}, Best Solution: {top_individual}, Fitness: {top_individual.fitness.values[0]}")
    print(f"ss=", np.round(sidespin_avg, 3),
          ", vs=", np.round(vertspin_avg, 3),
          ", speed=", np.round(cuespeed_avg, 3),
          ", cut=", np.round(cutangle_avg, 3),
          ", incline=", np.round(cueincline_avg, 3))
