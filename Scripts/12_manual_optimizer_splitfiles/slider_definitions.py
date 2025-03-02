# slider_definitions.py

from tkinter import Scale, HORIZONTAL

# Ball-ball parameter sliders
def create_ballball_sliders(slider_frame, ballball_hit_params, update_plot):
    ballball_a_slider = Scale(slider_frame, from_=0, to=0.02, resolution=0.0001, orient=HORIZONTAL, label="Ball-Ball a", length=400, command=update_plot)
    ballball_a_slider.set(ballball_hit_params['a'])
    ballball_a_slider.pack()

    ballball_b_slider = Scale(slider_frame, from_=0, to=0.2, resolution=0.001, orient=HORIZONTAL, label="Ball-Ball b", length=400, command=update_plot)
    ballball_b_slider.set(ballball_hit_params['b'])
    ballball_b_slider.pack()

    ballball_c_slider = Scale(slider_frame, from_=0, to=2, resolution=0.01, orient=HORIZONTAL, label="Ball-Ball c", length=400, command=update_plot)
    ballball_c_slider.set(ballball_hit_params['c'])
    ballball_c_slider.pack()

    return ballball_a_slider, ballball_b_slider, ballball_c_slider

# Physics parameter sliders
def create_physics_sliders(slider_frame, physics_params, update_plot):
    physics_u_slide_slider = Scale(slider_frame, from_=0, to=1, resolution=0.001, orient=HORIZONTAL, label="Physics u_slide", length=400, command=update_plot)
    physics_u_slide_slider.set(physics_params['u_slide'])
    physics_u_slide_slider.pack()

    physics_u_roll_slider = Scale(slider_frame, from_=0, to=0.1, resolution=0.001, orient=HORIZONTAL, label="Physics u_roll", length=400, command=update_plot)
    physics_u_roll_slider.set(physics_params['u_roll'])
    physics_u_roll_slider.pack()

    physics_u_sp_prop_slider = Scale(slider_frame, from_=0, to=1, resolution=0.01, orient=HORIZONTAL, label="Physics u_sp_prop", length=400, command=update_plot)
    physics_u_sp_prop_slider.set(physics_params['u_sp_prop'])
    physics_u_sp_prop_slider.pack()

    physics_e_ballball_slider = Scale(slider_frame, from_=0, to=1, resolution=0.001, orient=HORIZONTAL, label="Physics e_ballball", length=400, command=update_plot)
    physics_e_ballball_slider.set(physics_params['e_ballball'])
    physics_e_ballball_slider.pack()

    physics_e_cushion_slider = Scale(slider_frame, from_=0, to=1, resolution=0.001, orient=HORIZONTAL, label="Physics e_cushion", length=400, command=update_plot)
    physics_e_cushion_slider.set(physics_params['e_cushion'])
    physics_e_cushion_slider.pack()

    physics_f_cushion_slider = Scale(slider_frame, from_=0, to=1, resolution=0.001, orient=HORIZONTAL, label="Physics f_cushion", length=400, command=update_plot)
    physics_f_cushion_slider.set(physics_params['f_cushion'])
    physics_f_cushion_slider.pack()

    return physics_u_slide_slider, physics_u_roll_slider, physics_u_sp_prop_slider, physics_e_ballball_slider, physics_e_cushion_slider, physics_f_cushion_slider
