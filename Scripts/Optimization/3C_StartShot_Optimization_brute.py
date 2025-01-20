#! /usr/bin/env python
import pooltool as pt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
from scipy.optimize import differential_evolution
from scipy.optimize import dual_annealing
from scipy.optimize import brute
from pooltool.ruleset.three_cushion import is_point

start_time = time.time()

# We need a table, some balls, and a cue stick
# table = pt.Table.default("billiard")



def my_function(vars, system, sidespin_delta, vertspin_delta, cuespeed_delta, phi_delta, shotnums, Rball):

    sidespin_avg, vertspin_avg, cuespeed_avg, phi_avg = vars

    # Initialize an empty list to store angles
    points = np.zeros(shotnums)
    sidespin = sidespin_avg + sidespin_delta
    vertspin = vertspin_avg + vertspin_delta
    cuespeed = cuespeed_avg + cuespeed_delta
    phi = phi_avg + phi_delta

    for i in range(shotnums):
        points[i] = 0
        # check if shot is outside of squirt limit. If so, no point
        if 0.5**2 >= (sidespin.item(i)**2 + vertspin.item(i)**2): # This will ensure R^2 - a^2 - b^2 >= 0
            system.cue.set_state(a=sidespin.item(i), b=vertspin.item(i), V0=cuespeed.item(i), phi=phi.item(i))
            system.reset_balls()

            # Evolve the shot.
            pt.simulate(system, inplace=True)

            points[i] = 1 if is_point(system) else 0

    success = np.sum(points)/shotnums
    print(f"ss=",sidespin_avg, ", vs=",vertspin_avg, ", speed=",cuespeed_avg, ", phi=",phi_avg, ", success=",success*100)
    return 1-success


# Ball Positions
wpos = (0.5275, 0.71)  # White
ypos = (0.71, 0.71)  # Yellow
rpos = (0.71, 2.13)  # Red

# define the properties
u_slide = 0.15
u_roll = 0.005
u_sp_prop = 10 * 2 / 5 / 9
u_ballball = 0.05
e_ballball = 0.95
e_cushion = 0.9
f_cushion = 0.15
grav = 9.81

mball = 0.210
Rball = 61.5 / 1000 / 2

cue_mass = 0.576
cue_len = 1.47
cue_tip_R = 0.022
cue_tip_mass = 0.0000001

# Build a table with default BILLIARD specs
table = pt.Table.default(pt.TableType.BILLIARD)

# create the cue
cue_specs = pt.objects.CueSpecs(
    brand="Predator",
    M=cue_mass,
    length=cue_len,
    tip_radius=cue_tip_R,
    butt_radius=0.02,
    end_mass=cue_tip_mass,
)
cue = pt.Cue(cue_ball_id="white", specs=cue_specs)

# Generate the ball layout from the THREECUSHION GameType using the BILLIARD table
# balls = pt.get_rack(pt.GameType.THREECUSHION, table=table)

# Create balls
wball = pt.Ball.create("white", xy=wpos, m=mball, R=Rball,
                       u_s=u_slide, u_r=u_roll, u_sp_proportionality=u_sp_prop, u_b=u_ballball,
                       e_b=e_ballball, e_c=e_cushion,
                       f_c=f_cushion, g=grav)

yball = pt.Ball.create("yellow", xy=ypos, m=mball, R=Rball,
                       u_s=u_slide, u_r=u_roll, u_sp_proportionality=u_sp_prop, u_b=u_ballball,
                       e_b=e_ballball, e_c=e_cushion,
                       f_c=f_cushion, g=grav)

rball = pt.Ball.create("red", xy=rpos, m=mball, R=Rball,
                       u_s=u_slide, u_r=u_roll, u_sp_proportionality=u_sp_prop, u_b=u_ballball,
                       e_b=e_ballball, e_c=e_cushion,
                       f_c=f_cushion, g=grav)

# Wrap it up as a System
system_template = pt.System(
    table=table,
    balls=(wball, yball, rball),
    cue=cue,
)

# Creates a deep copy of the template
system = system_template.copy()

# shot props
sidespin_stddev = 0.05
vertspin_stddev = 0.05
cuespeed_stddev = 0.2
phi_delta_stddev = 0.1
shotnums = 250

# generate shot props from new mean values
sidespin_delta = np.random.normal(loc=0, scale=sidespin_stddev, size=shotnums)
vertspin_delta = np.random.normal(loc=0, scale=vertspin_stddev, size=shotnums)
cuespeed_delta = np.random.normal(loc=0, scale=cuespeed_stddev, size=shotnums)
phi_delta = np.random.normal(loc=0, scale=phi_delta_stddev, size=shotnums)

phi = pt.aim.at_ball(system, "red", cut=35)
#
# sidespin = np.linspace(0, 0.5, 5)
# vertspin = np.linspace(0, 0.5, 5)
# cuespeed = np.linspace(2, 5, 5)
# phi = np.linspace(phi-1, phi+1, 5)
# bounds = ((0.0, 0.5, 5),(0.0, 0.5, 5),(2.0, 5.0, 3),(phi-1, phi+1, 5))

# sidespin = slice(0, 0.5, 0.1)
# vertspin = slice(0, 0.5, 0.1)
# cuespeed = slice(2, 5, 1)
# phi = slice(phi-1, phi+1, 0.25)
# bounds = np.mgrid[sidespin, vertspin, cuespeed, phi]
# bounds = ((0, 0.5, 3), (0.0, 0.5, 3), (2, 5, 3), (phi-1, phi+1, 3))

bounds = ((-0.5, 0.5), (-0.5, 0.5), (2, 7), (phi-1, phi+1))


# Use scipy.optimize.minimize to optimize only a, b, c
# result = differential_evolution(
#     my_function,
#     bounds,
#     popsize = 500,
#     mutation = 0.8,
#     recombination = 0.5,
#     disp = True,
#     seed = 42,
#     strategy = 'best1bin',
#     args = (system, sidespin_delta, vertspin_delta, cuespeed_delta, phi_delta, shotnums, Rball))

# result = dual_annealing(
#     my_function,
#     bounds,
#     args=(system, sidespin_delta, vertspin_delta, cuespeed_delta, phi_delta, shotnums, Rball))

result = brute(
    my_function,
    ranges=bounds,
    Ns = 5,
    args=(system, sidespin_delta, vertspin_delta, cuespeed_delta, phi_delta, shotnums, Rball),
    full_output=True)


end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time} seconds")

# Print the result
print(result[0])
print(result[1])
# print(f"Minimum, {1-result.fun}, found at sidespin = {result.x[0]}, topspin = {result.x[1]}, speed = {result.x[2]}, phi = {result.x[3]}")
