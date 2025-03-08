# slider_definitions.py

from tkinter import Scale, HORIZONTAL

# Shot parameter sliders
def create_sliders(slider_frame, params, shots_actual, update_shot_id, update_plot):
    # Shot parameter sliders
    slider_length = 400
    slider_height = 30  

    sliders = {}

    # Shot selector slider
    sliders['shot_id'] = shot_id_slider = Scale(slider_frame, from_=0, to=len(shots_actual) - 1, orient=HORIZONTAL, label="Shot", length=slider_length, command=update_shot_id)
    shot_id_slider.set(0)
    shot_id_slider.pack()

    sliders['shot_a'] = shot_a_slider = Scale(slider_frame, from_=-0.6, to=0.6, resolution=0.001, orient=HORIZONTAL, label="Shot a", length=slider_length, command=update_plot)
    shot_a_slider.set(params['shot_a'])
    shot_a_slider.pack()

    sliders['shot_b'] = shot_b_slider = Scale(slider_frame, from_=-0.6, to=0.6, resolution=0.001, orient=HORIZONTAL, label="Shot b", length=slider_length, command=update_plot)
    shot_b_slider.set(params['shot_b'])
    shot_b_slider.pack()

    sliders['shot_phi'] = shot_phi_slider = Scale(slider_frame, from_=-180, to=180, resolution=0.01, orient=HORIZONTAL, label="Shot phi", length=slider_length, command=update_plot)
    shot_phi_slider.set(params['shot_phi'])
    shot_phi_slider.pack()

    sliders['shot_v'] = shot_v_slider = Scale(slider_frame, from_=0, to=10, resolution=0.1, orient=HORIZONTAL, label="Shot v", length=slider_length, command=update_plot)
    shot_v_slider.set(params['shot_v'])
    shot_v_slider.pack()

    sliders['shot_theta'] = shot_theta_slider = Scale(slider_frame, from_=0, to=90, resolution=0.1, orient=HORIZONTAL, label="Shot theta", length=slider_length, command=update_plot)
    shot_theta_slider.set(params['shot_theta'])
    shot_theta_slider.pack()

    # Ball-ball parameter sliders
    sliders['ballball_a'] = ballball_a_slider = Scale(slider_frame, from_=0, to=0.1, resolution=0.01, orient=HORIZONTAL, label="Ball-Ball a", length=slider_length, command=update_plot)
    ballball_a_slider.set(params['ballball_a'])
    ballball_a_slider.pack()

    sliders['ballball_b'] = ballball_b_slider = Scale(slider_frame, from_=0, to=1.0, resolution=0.01, orient=HORIZONTAL, label="Ball-Ball b", length=slider_length, command=update_plot)
    ballball_b_slider.set(params['ballball_b'])
    ballball_b_slider.pack()

    sliders['ballball_c'] = ballball_c_slider = Scale(slider_frame, from_=0, to=5, resolution=0.1, orient=HORIZONTAL, label="Ball-Ball c", length=slider_length, command=update_plot)
    ballball_c_slider.set(params['ballball_c'])
    ballball_c_slider.pack()

    # Physics parameter sliders
    sliders['physics_u_slide'] = physics_u_slide_slider = Scale(slider_frame, from_=0, to=1, resolution=0.001, orient=HORIZONTAL, label="Physics u_slide", length=slider_length, command=update_plot)
    physics_u_slide_slider.set(params['physics_u_slide'])
    physics_u_slide_slider.pack()

    sliders['physics_u_roll'] = physics_u_roll_slider = Scale(slider_frame, from_=0, to=0.1, resolution=0.001, orient=HORIZONTAL, label="Physics u_roll", length=slider_length, command=update_plot)
    physics_u_roll_slider.set(params['physics_u_roll'])
    physics_u_roll_slider.pack()

    sliders['physics_u_sp_prop'] = physics_u_sp_prop_slider = Scale(slider_frame, from_=0, to=1, resolution=0.01, orient=HORIZONTAL, label="Physics u_sp_prop", length=slider_length, command=update_plot)
    physics_u_sp_prop_slider.set(params['physics_u_sp_prop'])
    physics_u_sp_prop_slider.pack()

    sliders['physics_e_ballball'] = physics_e_ballball_slider = Scale(slider_frame, from_=0, to=1, resolution=0.001, orient=HORIZONTAL, label="Physics e_ballball", length=slider_length, command=update_plot)
    physics_e_ballball_slider.set(params['physics_e_ballball'])
    physics_e_ballball_slider.pack()

    sliders['physics_e_cushion'] = physics_e_cushion_slider = Scale(slider_frame, from_=0, to=1, resolution=0.001, orient=HORIZONTAL, label="Physics e_cushion", length=slider_length, command=update_plot)
    physics_e_cushion_slider.set(params['physics_e_cushion'])
    physics_e_cushion_slider.pack()

    sliders['physics_f_cushion'] = physics_f_cushion_slider = Scale(slider_frame, from_=0, to=1, resolution=0.001, orient=HORIZONTAL, label="Physics f_cushion", length=slider_length, command=update_plot)
    physics_f_cushion_slider.set(params['physics_f_cushion'])
    physics_f_cushion_slider.pack()

    sliders['physics_h_cushion'] = physics_cushion_height_slider = Scale(slider_frame, from_=0.035, to=0.039, resolution=0.0001, orient=HORIZONTAL, label="Physics h_cushion", length=slider_length, command=update_plot)
    physics_cushion_height_slider.set(params['physics_h_cushion'])
    physics_cushion_height_slider.pack()

    return sliders

