import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
import os
import time

# class for archimedial spiral of circular or rectangular cross section
class Coil: 
    # ctor
    def __init__(self, x_shift, y_shift, z_shift, init_rad, a, turns_per_layer, seg_per_turn,  cross_sec = 'circular', 
                 wire_radius = 0.001, wire_height = 0, wire_width = 0, layers = 1, layer_distance = 0, theta_direction = 'right_hand', 
                 layer_direction = 'opposite', rot_x = 0, rot_y = 0, rot_z = 0) :
        # Inputs: 
        # x_shift, y_shift, z_shift: shifts in x, y, z axes
        # init_rad: initial radius
        # a: growth rate / pitch (distance between consecturive turns)
        # turns_per_layer: number of turns in each layer, must be integer
        # seg_per_turn: number of segments in each turn, must be integer
        # cross_sec: cross section shape (circular or rectangular) (default is circular)
        # wire_radius: wire radius for circular cross section (default is 1mm)
        # wire_length: wire height for rectangular cross section (default is 0)
        # wire_width: wire width for rectangular cross section (default is 0)
        # layers: number of layers, must be integer (default is 1)
        # layer_distance: distance between layers (default is 0 because default is 1 layer)
        # theta_direction: direction of theta increase (right_hand or left_hand) (default is right hand)
        # layer_direction: direction of theta increase for adjectent layers increase (opposite or same) (default is opposite)
        # rot_x: rotation about the x axis (default is 0)
        # rot_y: rotation about the y axis (default is 0)
        # rot_z: rotation about the z axis (equivalent to a phase shift) (default is 0)
        
        # set members
        self.x_shift = x_shift
        self.y_shift = y_shift
        self.z_shift = z_shift
        self.init_rad = init_rad
        self.a = a
        self.turns_per_layer = turns_per_layer
        self.seg_per_turn = seg_per_turn
        self.cross_sec = cross_sec
        self.wire_radius = wire_radius
        self.wire_height = wire_height
        self.wire_width = wire_width
        self.layers = layers
        self.layer_distance = layer_distance
        self.theta_direction = theta_direction
        self.layer_direction = layer_direction
        self.rot_x = rot_x
        self.rot_y = rot_y
        self.rot_z = rot_z

       
        # generate the coordinates
        #self.coords = self.make_layer(x_shift, y_shift, z_shift, init_rad, a, turns_per_layer, seg_per_turn, theta_direction, rot_x, rot_y, rot_z)
        self.make_coil()

    # helper function to make one layer of a coil
    def make_layer(self, x_shift, y_shift, z_shift, init_rad, a, turns_per_layer, seg_per_turn, theta_direction, rot_x, rot_y, rot_z) :
        # generate theta values        
        # right now i am skipping layers...
        theta = np.linspace(rot_z, rot_z + (2 * np.pi * turns_per_layer), turns_per_layer * seg_per_turn + 1)

        # Calculate the radial distance for each theta
        R = init_rad + (a / (2 * np.pi)) * theta

        # convert to cartesian
        x = R * np.cos(theta)
        y = R * np.sin(theta)
        z = np.zeros_like(x)
        coords = np.array([x, y, z])

        # make right hand or left hand (flip across x-axis if left hand)
        if theta_direction == 'left_hand':
            # reflect across x-axis
            coords[1, :] = -coords[1, :]
        elif theta_direction != 'right_hand':
            raise ValueError('Theta direction must be right_hand or left_hand')

        # make rotation matrices and apply if necessary
        if rot_x != 0:
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(rot_x), np.sin(rot_x)],
                [0, -np.sin(rot_x), np.cos(rot_x)]])
            coords = rotation_matrix @ coords
        if rot_y != 0:
            rotation_matrix = np.array([
                [np.cos(rot_y), 0, -np.sin(rot_y)],
                [0, 1, 0],
                [np.sin(rot_y), 0, np.cos(rot_y)]])
            coords = rotation_matrix @ coords
        
        # shift the coordinates
        coords[0, :] += x_shift
        coords[1, :] += y_shift
        coords[2, :] += z_shift

        return coords
    
    # function to make a coil with one layer or multiple layers
    def make_coil(self) :
        # make an empty np array to store the coordinates
        coords = np.empty((3, 0))
        
        for i in range(self.layers) :
            # if i is even or 0, the coil goes inside to outside and the direction will be the same regardless of layer number
            # if i is odd, the coil goes outside to inside and direction may change
            if i % 2 == 0 :
                # radius starts from the inside
                radius = self.init_rad
                a = self.a
                direction = self.theta_direction
            else :
                # radius starts from the outside
                radius = self.init_rad + self.a * self.turns_per_layer
                a = -self.a
                if self.layer_direction == 'opposite' :
                    if self.theta_direction == 'right_hand' :
                        direction = 'left_hand'
                    else :
                        direction = 'right_hand'
                else : 
                    direction = self.theta_direction

            # make a layer
            layer = self.make_layer(self.x_shift, self.y_shift, self.z_shift + i * self.layer_distance, 
                                    radius, a, self.turns_per_layer, self.seg_per_turn, 
                                    direction, self.rot_x, self.rot_y, self.rot_z) 
            
            # add the layer to the coordinates
            coords = np.concatenate((coords, layer), axis=1)
        
        # set the coordinates
        self.coords = coords

 
# function that plots each coil in the list of coils
def plot_coils(coils, file_path=None) :
    # supress output for now
    plt.ioff()

    # add set axis length
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for ind, coil in enumerate(coils) :
        # Plot coil
        label_str = f"Coil {ind + 1}"
        # make the first one blue and second red
        if ind == 0 :
            ax.plot(coil.coords[0, :], coil.coords[1, :], coil.coords[2, :], label=label_str, color='blue')
        elif ind == 1 :
            ax.plot(coil.coords[0, :], coil.coords[1, :], coil.coords[2, :], label=label_str, color='red')
        else :
            ax.plot(coil.coords[0, :], coil.coords[1, :], coil.coords[2, :], label=label_str)
        
        # find the max value for the limit
        if ind == 0 :
            lim = coil.init_rad + coil.a * coil.turns_per_layer
        else :
            if coil.init_rad + coil.a * coil.turns_per_layer > lim :
                lim = coil.init_rad + coil.a * coil.turns_per_layer

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Coil(s) Visualization')
    ax.legend()

    # set axis limits
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    if file_path :
        plt.savefig(file_path, format='jpg')
    else :
        plt.show()

    # Clear the figure (optional) for ipynb
    plt.clf()
    plt.close(fig)


# function to generate coils from a txt file
def generate_coils_from_txt(file_path):
    coils = []
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
            # Find the line starting with "Number of coil(s):"
            num_coils = 0
            for line in lines:
                if line.lower().startswith("number of coil".lower()):
                    num_coils = int(line.split(":")[1].strip())
                    break
            else:
                # If we don't find the line, raise an error or handle the case
                raise ValueError("Number of coil(s) not specified in the file.")
            
            # Skip lines until the coil data starts
            data_lines = iter(lines[lines.index(line) + 1:])
            coil_count = 0
            started = False # boolean to check if we have started reading a coil data
            while coil_count < num_coils:
                try:
                    while True: # read data in and check if it is a valid coil data
                        line = next(data_lines).strip()
                        if (line.startswith("-") and not started): # Skip separator lines
                            started = True # change boolean to True when we start reading coil data
                            continue
                        if line.lower().startswith("x-shift:".lower()):
                            x_shift = float(line.split(":")[1].strip()) / 100 # convert cm to m
                            if not isinstance(x_shift, (int, float)):
                                raise TypeError("x_shift must be a number.")
                        elif line.lower().startswith("y-shift:".lower()):
                            y_shift = float(line.split(":")[1].strip()) / 100 # convert cm to m
                            if not isinstance(y_shift, (int, float)):
                                raise TypeError("y_shift must be a number.")
                        elif line.lower().startswith("z-shift:".lower()):
                            z_shift = float(line.split(":")[1].strip()) / 100 # convert cm to m
                            if not isinstance(z_shift, (int, float)):
                                raise TypeError("z_shift must be a number.")
                        elif line.lower().startswith("initial radius:".lower()):
                            init_rad = float(line.split(":")[1].strip()) / 100 # convert cm to m
                            if not isinstance(init_rad, (int, float)) or init_rad <= 0:
                                raise ValueError("init_rad must be a positive number.")
                        elif line.lower().startswith("pitch/coil progression:".lower()):
                            a = float(line.split(":")[1].strip()) / 100 # convert cm to m
                            if not isinstance(a, (int, float)) or a <= 0:
                                raise ValueError("a must be a positive number.")
                        elif line.lower().startswith("turns per layer:".lower()):
                            turns_per_layer = int(line.split(":")[1].strip())
                            if not isinstance(turns_per_layer, int) or turns_per_layer <= 0:
                                raise ValueError("turns_per_layer must be a positive integer.")
                        elif line.lower().startswith("segments per turn:".lower()):
                            seg_per_turn = int(line.split(":")[1].strip())
                            if not isinstance(seg_per_turn, int) or seg_per_turn <= 0:
                                raise ValueError("seg_per_turn must be a positive integer.")
                        elif line.lower().startswith("layers:".lower()):
                            layers = int(line.split(":")[1].strip())
                            if not isinstance(layers, int) or layers <= 0:
                                raise ValueError("layers must be a positive integer.")
                        elif line.lower().startswith("distance between layers:".lower()):
                            layer_distance = float(line.split(":")[1].strip()) / 100 # convert cm to m
                            if not isinstance(layer_distance, (int, float)):
                                raise ValueError("layer_distance must be a number.")
                        elif line.lower().startswith("cross section type".lower()):
                            cross_sec = line.split(":")[1].strip().lower()
                            if cross_sec not in ('circular', 'rectangular'):
                                raise ValueError("cross_sec must be 'circular' or 'rectangular'.")
                        elif line.lower().startswith("cross sectional wire radius:".lower()):
                            wire_radius = float(line.split(":")[1].strip()) / 100 # convert cm to m
                            if cross_sec == 'circular':
                                if not isinstance(wire_radius, (int, float)) or wire_radius <= 0:
                                    raise ValueError("wire_radius must be a positive number for circular cross section.")
                        elif line.lower().startswith("cross sectional wire height:".lower()):
                            wire_height = float(line.split(":")[1].strip()) / 100 # convert cm to m
                            if cross_sec == 'rectangular':
                                if not isinstance(wire_height, (int, float)) or wire_height <= 0:
                                    raise ValueError("wire_height must be a positive number for rectangular cross section.")
                        elif line.lower().startswith("cross sectional wire width:".lower()):
                            wire_width = float(line.split(":")[1].strip()) / 100 # convert cm to m
                            if cross_sec == 'rectangular':
                                if not isinstance(wire_width, (int, float)) or wire_width <= 0:
                                    raise ValueError("wire_width must be a positive number for rectangular cross section.")
                        elif line.lower().startswith("spiral direction".lower()):
                            theta_direction = line.split(":")[1].strip()
                            if theta_direction not in ('right_hand', 'left_hand'):
                                raise ValueError("theta_direction must be 'right_hand' or 'left_hand'.")
                        elif line.lower().startswith("will layers be turning in the same or opposite directions".lower()):
                            layer_direction = line.split(":")[1].strip()
                            if layer_direction not in ('opposite', 'same'):
                                raise ValueError("layer_direction must be 'opposite' or 'same'.")
                        elif line.lower().startswith("rotation about x-axis:".lower()):
                            rot_x = float(line.split(":")[1].strip())
                            if not isinstance(rot_x, (int, float)):
                                raise TypeError("rot_x must be a number.")
                        elif line.lower().startswith("rotation about y-axis:".lower()):
                            rot_y = float(line.split(":")[1].strip())
                            if not isinstance(rot_y, (int, float)):
                                raise TypeError("rot_y must be a number.")
                        elif line.lower().startswith("rotation about z-axis/phase shift:".lower()):
                            rot_z = float(line.split(":")[1].strip())
                            if not isinstance(rot_z, (int, float)):
                                raise TypeError("rot_z must be a number.")
                        elif line.startswith("-") and started:
                            # switch started back to True to indicate we are done reading the coil data and are begining to read the next coil data
                            started = True
                            break

                    # Create a Coil2 instance with the parsed parameters
                    coil = Coil(x_shift, y_shift, z_shift, init_rad, a, turns_per_layer, seg_per_turn, 
                                 cross_sec, wire_radius, wire_height, wire_width, layers, layer_distance, 
                                 theta_direction, layer_direction, rot_x, rot_y, rot_z)
                    coils.append(coil)
                    coil_count += 1


                except StopIteration:
                    print("Unexpected end of file or missing data for coils.")
                    break

    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except ValueError as e:
        print(f"Error parsing data: {e}")

    return coils


### Functions for inductance calculations
# partial self inductance for circular cross section
def L_part_hubert(points, r_w) :
    mu_o = 4 * np.pi * 10**-7
    l = np.linalg.norm(points[:, 1] - points[:, 0])
    L = (mu_o / (2 * np.pi)) * (
        l * np.log(np.sqrt(l**2 + r_w*2) + l) - l * (np.log(r_w) - 0.25)
        - np.sqrt(l**2 + r_w**2) + 0.905415*r_w
    )
    return L

# partial self inductance for rectangular cross section
def L_part_rect_cross(points, w, t) :
    mu_o = 4 * np.pi * 10**-7
    l = np.linalg.norm(points[:, 1] - points[:, 0])
    L = (mu_o / (2 * np.pi)) * l * (np.log((2 * l) / (w + t)) + 0.5)
    return L

# helper function
# assumes lengths in cm
def equal_parallel(l, d) : 
    M = 0.002 * l * (
        np.log((l/d) + np.sqrt(1 + (l/d)**2)) - np.sqrt(1 + (d/l)**2) + d/l
    )
    return M * 10**-6 # convert to H

# if parallel
# assumes lengths in cm
def parralel_lines(A, B, C, D) : 
    threshold = 1e-6 # threshold value for rounding errors
    # check if exactly equal
    # for this, AC is perpendicular to AB and CD and l = m
    AB = B - A
    CD = D - C
    CA = A - C
    CB = B - C
    DA = A - D
    DB = B - D
    l = np.linalg.norm(AB)
    m = np.linalg.norm(CD)
    # calculate the distance between
    CA_angle = np.arccos(np.dot(CA, CD) / (np.linalg.norm(CA) * m)) # angle between CA and CD
    DA_angle = np.arccos(np.dot(DA, CD) / (np.linalg.norm(DA) * m)) # angle between DA and CD
    CB_angle = np.arccos(np.dot(CB, CD) / (np.linalg.norm(CB) * m)) # angle betwwen CB and CD
    DB_angle = np.arccos(np.dot(DB, CD) / (np.linalg.norm(DB) * m)) # angle between DB and CD
    d = np.linalg.norm(CA) * np.sin(CA_angle)

    exact_bool = (np.abs(l - m) < threshold) and ((np.abs(np.pi/2 - CA_angle) < threshold) or (np.abs(np.pi/2 - CB_angle) < threshold))
    parallel_bool = np.abs((np.dot(CD, AB) / (l * m)) - 1) < threshold # true if AB and CD are parallel
    anti_parallel_bool = np.abs((np.dot(CD, AB) / (l * m)) + 1) < threshold # true if AB and CD are anti-parallel
    no_overlap_parallel_bool = ((CB_angle > np.pi/2) or (DA_angle < np.pi/2)) and (parallel_bool) # true if DA angle is acute or CB angle is obtuse and must be parallel
    no_overlap_anti_parallel_bool = ((CA_angle > np.pi/2) or (DB_angle < np.pi/2)) and (anti_parallel_bool) # true if DB angle is acute or CA angle is obtuse and must be anti-parallel
    starting_end_parallel_bool = (np.abs(np.pi/2 - CA_angle) < threshold) and ~(exact_bool) and parallel_bool # true if starting ends are common perpendicular. if not exact bool, and CA is pi/2, then true
    starting_end_anti_parallel_bool = (np.abs(np.pi/2 - CB_angle) < threshold) and ~(exact_bool) and anti_parallel_bool # true if starting ends are common perpendicular. if not exact bool, and CB is pi/2, then true anti parallel
    straight_line_bool = (np.linalg.norm(np.cross(CA, CD)) < threshold)# CA is parallel with CD
    opp_end_perp_parallel_bool = (np.abs(np.pi/2 - DA_angle) < threshold) and parallel_bool # true if opposite ends are common perpendicular and parallel
    opp_end_perp_anti_parallel_bool = (np.abs(np.pi/2 - DB_angle) < threshold) and anti_parallel_bool # true if opposite ends are common perpendicular and anti parallel
    inside_other_parallel_bool = (((CA_angle < np.pi/2) and (DB_angle > np.pi/2)) or ((CA_angle > np.pi/2) and (DB_angle < np.pi/2))) and parallel_bool # if one is inside the other and parallel
    inside_other_anti_parallel_bool = (((CB_angle > np.pi/2) and (DA_angle < np.pi/2)) or ((CB_angle < np.pi/2) and (DA_angle > np.pi/2)))and anti_parallel_bool # if one is inside the other and anti-parallel
    overlapping_parallel_bool = (((CA_angle < np.pi/2) and (DB_angle < np.pi/2)) or ((CA_angle > np.pi/2) and (DB_angle > np.pi/2))) and parallel_bool and ~(no_overlap_parallel_bool or no_overlap_anti_parallel_bool) # overlapping filaments (Both acute or both obutse) and parallel and not no overlap
    overlapping_anti_parallel_bool  = (((CB_angle < np.pi/2) and (DA_angle < np.pi/2)) or ((CB_angle > np.pi/2) and (DA_angle > np.pi/2))) and anti_parallel_bool and ~(no_overlap_parallel_bool or no_overlap_anti_parallel_bool) # overlapping filaments (Both acute or both obutse) and anti-parallel and not no overlap
    if exact_bool :
        return equal_parallel(l, d) * (np.dot(AB, CD) / (l *m)) # multiply by cos of angle between the two to get sign correct (its either -1 or 1)
    elif straight_line_bool : # check for straight line before no overlap
        # check if touch
        if (np.linalg.norm(CB) < threshold) or (np.linalg.norm(DA) < threshold) : # if C and B are the same or D and A are the same
            M = 0.001 * (
                l * np.log((l + m) / l) + m * np.log((l + m) / m)
            ) # mutual inductance calculation
            return M * 10**-6 # convert to H
        else :
            # calculate distance between the two
            delta = min(np.linalg.norm(CB), np.linalg.norm(DA), np.linalg.norm(DB))
            M = 0.001 * (
                (l + m + delta) * np.log(l + m + delta)
                - (l + delta) * np.log(l + delta)
                - (m + delta) * np.log(m + delta)
                + delta * np.log(delta)
            )
            return M * 10**-6 * (np.dot(CA, CD) / (np.linalg.norm(CA) * m)) # multiply by cos of angle between the two to get sign correct (its either -1 or 1)
    elif no_overlap_parallel_bool : # might be better to use this an the final else case....
        # calculate parallel distance between the two
        # if CB angle is obtuse
        if CB_angle > np.pi/2 :
            delta = np.linalg.norm(CB) * np.cos(np.pi - CB_angle)
        else :
            delta = np.linalg.norm(DA) * np.cos(DA_angle)
        # calculate mutual inductance through summation of equal parallel lines
        M = 0.5 * (
            (equal_parallel(l + m + delta,d) + equal_parallel(delta, d)) - (equal_parallel(l + delta, d) + equal_parallel(m + delta, d))
        )
        return M
    elif no_overlap_anti_parallel_bool :
        # calculate parallel distance between the two
        # if CB angle is obtuse
        if CA_angle > np.pi/2 :
            delta = np.linalg.norm(CA) * np.cos(np.pi - CA_angle)
        else :
            delta = np.linalg.norm(DB) * np.cos(DB_angle)
        # calculate mutual inductance through summation of equal parallel lines
        M = 0.5 * (
            (equal_parallel(l + m + delta,d) + equal_parallel(delta, d)) - (equal_parallel(l + delta, d) + equal_parallel(m + delta, d))
        )
        return M * -1 # multiply by -1 to get the sign correct
    elif starting_end_parallel_bool :
        M = 0.5 * (
            equal_parallel(l, d) 
            + equal_parallel(m, d)
            - equal_parallel(np.abs(l - m), d)
        )
        return M # already converted to H
    elif starting_end_anti_parallel_bool :
        M = 0.5 * (
            equal_parallel(l, d) 
            + equal_parallel(m, d)
            - equal_parallel(np.abs(l - m), d)
        )
        return M * -1
    elif opp_end_perp_parallel_bool :
        M = 0.5 * (
            equal_parallel(l + m, d) 
            - equal_parallel(l, d)
            - equal_parallel(m, d)
        )
        return M
    elif opp_end_perp_anti_parallel_bool :
        M = 0.5 * (
            equal_parallel(l + m, d) 
            - equal_parallel(l, d)
            - equal_parallel(m, d)
        )
        return M * -1
    elif inside_other_parallel_bool :
        # calculate p and q first
        p = np.abs(np.cos(CA_angle)) * np.linalg.norm(CA)
        q = np.abs(np.cos(DB_angle)) * np.linalg.norm(DB)
        # get min length to use
        side = min(m, l)
        # calculate mutual inductance
        M = 0.5 * (
            equal_parallel(side + p, d)
             + equal_parallel(side + q, d)
             - equal_parallel(p, d)
             - equal_parallel(q, d)
        )
        return M
    elif inside_other_anti_parallel_bool :
        # calculate p and q first
        p = np.abs(np.cos(CB_angle)) * np.linalg.norm(CB)
        q = np.abs(np.cos(DA_angle)) * np.linalg.norm(DA)
        # get min length to use
        side = min(m, l)
        # calculate mutual inductance
        M = 0.5 * (
            equal_parallel(side + p, d)
             + equal_parallel(side + q, d)
             - equal_parallel(p, d)
             - equal_parallel(q, d)
        )
        return M * -1 # multiply by -1 to get the sign correct
    elif overlapping_parallel_bool :
        delta = np.abs(np.cos(CA_angle)) * np.linalg.norm(CA) 
        M = 0.5 * (
            equal_parallel(l + m - delta, d)
            + equal_parallel(delta, d)
            - equal_parallel(l - delta, d)
            - equal_parallel(m - delta, d)
        )
        return M
    elif overlapping_anti_parallel_bool :
        delta = np.abs(np.cos(CB_angle)) * np.linalg.norm(CB) 
        M = 0.5 * (
            equal_parallel(l + m - delta, d)
            + equal_parallel(delta, d)
            - equal_parallel(l - delta, d)
            - equal_parallel(m - delta, d)
        )
        return M * -1 # multiply by -1 to get the sign correct
    else : # should never happen. But I included this as to not cause an error. Maybe I should raise an error instead or return NaN
        return np.nan

### coplanar cases
def coplanar_lines(A, B, C, D) : 
    threshold = 1e-6 # threshold value for rounding errors
    # assume lengths in cm
    AB = B - A
    CD = D - C
    l = np.linalg.norm(AB)
    m = np.linalg.norm(CD)

    # check for intersecting
    meet_bool = (np.linalg.norm(A - C) < threshold) or (np.linalg.norm(B - C) < threshold) or (np.linalg.norm(B - D) < threshold) or (np.linalg.norm(A- D) < threshold)
    same_length_bool = np.abs(l - m) < threshold

    if meet_bool and same_length_bool :
        # find distance between the ends
        if np.linalg.norm(A - C) < threshold :
            R1 = np.linalg.norm(B - D)
        elif np.linalg.norm(B - C) < threshold :
            R1 = np.linalg.norm(A - D)
        elif np.linalg.norm(B - D) < threshold :
            R1 = np.linalg.norm(A - C)
        else :
            R1 = np.linalg.norm(B - C)
        cos_eps = 1 - (R1**2 / (2 * l**2))
        M = 0.004 * l * cos_eps * np.arctanh(l / (l + R1))
        return M * 10**-6 * -1 # convert to H account for sign
    elif meet_bool and ~same_length_bool :
        # find distance between the ends
        if np.linalg.norm(A - C) < threshold :
            R = np.linalg.norm(B - D)
        elif np.linalg.norm(B - C) < threshold :
            R = np.linalg.norm(A - D)
        elif np.linalg.norm(B - D) < threshold :
            R = np.linalg.norm(A - C)
        else :
            R = np.linalg.norm(B - C)
        cos_eps = (l**2 + m**2 - R**2) / (2 * l * m)
        check = np.dot(AB, CD) / (l * m)
        phi = np.arccos(check) * 180 / np.pi
        M = 0.001 * l * cos_eps * (
            np.log((1 + (m/l) + (R/l))/ (1 - (m/l) +  (R/l)))
            + (m/l) * np.log(((m/l) + (R/l) + 1) / ((m/l) + (R/l) - 1)) 
        )
        return M * 10**-6 * -1# convert to H
    else : # if not meeting ..
        # check angle 
        theta = np.arccos(np.dot(AB, CD) / (l * m))

        # Handle cases now
        if theta < np.pi/2 : # acute
            # set sign change for later
            sign_change = 1
            # check which side is longer. only change if CA > DB. Sides should never be equal
            if np.linalg.norm(A - C) > np.linalg.norm(B - D) :
                C, D, A, B = D, C, B, A # swap
            # do nothing if CA < DB
        elif theta > np.pi/2 : # if obtuse (perpendicular and parallel case should never happen)
            # set sign change for later
            sign_change = -1
            # check to see which side is longer
            if np.linalg.norm(B - C) < np.linalg.norm(A - D) : # if CB < DA
                A, B = B, A
            elif np.linalg.norm(B - C) > np.linalg.norm(A - D) : # if CB > DA
                C, D = D, C
        
        # recalculate parameters
        AB = B - A
        CD = D - C
        l = np.linalg.norm(AB)
        m = np.linalg.norm(CD)
        R1 = np.linalg.norm(B - D)
        R2 = np.linalg.norm(B - C)
        R3 = np.linalg.norm(A - C)
        R4 = np.linalg.norm(A - D)

        # calculate other params
        alpha_sq = R4**2 - R3**2 + R2**2 - R1**2
        cos_eps_2 = alpha_sq / (l * m)
        mu = l * (
            2 * m**2 * (R2**2 - R3**2 - l**2) + alpha_sq * (R4**2 - R3**2 - m**2)
        ) / (4 * l**2 * m**2 - alpha_sq**2)
        nu = m * (
            2 * l**2 * (R4**2 - R3**2 - m**2) + alpha_sq * (R2**2 - R3**2 - l**2)
        ) / (4 * l**2 * m**2 - alpha_sq**2)
        M = cos_eps_2 * 0.001 * (
            (mu + l) * np.arctanh(m / (R1 + R2)) + (nu + m) * np.arctanh(l / (R1 + R4))
            - mu * np.arctanh(m / (R3 + R4)) - nu * np.arctanh(l / (R2 + R3))
        )
        return M * sign_change * 10**-6 # account for sign change and convert to H

# Any orientation case (not coplanar, parallel, nor perpendicular) via Grover 1972
def any_desired_position(A, B, a, b) :
    # for this, look at geometry to calculate partial mutual inductance, then check the sign
    # EVERYTHING SHOULD COME IN AS CM
    # make sure r3 is the smallest one. make appropriate changes if not
    Aa = np.linalg.norm(A - a)
    Ba = np.linalg.norm(B - a)
    Ab = np.linalg.norm(A - b)
    Bb = np.linalg.norm(B - b)
    min_one = np.min([Aa, Ba, Ab, Bb])

    if min_one == Aa :
        # do nothing
        sign = 1
    elif min_one == Bb :
        # flip them
        A, B, a, b = B, A, b, a
        sign = 1
    elif min_one == Ab :
        # flim a and b
        a, b = b, a
        sign = -1
    elif min_one == Ba :
        # flip A and B
        A, B = B, A
        sign = -1
    
    # recalculate...
    R1 = np.linalg.norm(B - b)
    R2 = np.linalg.norm(B - a)
    R3 = np.linalg.norm(A - a)
    R4 = np.linalg.norm(A - b)
    l = np.linalg.norm(B - A)
    m = np.linalg.norm(b - a)

    # calculate other params
    alpha_sq = R4**2 - R3**2 + R2**2 - R1**2
    cos_eps = alpha_sq / (2 * l * m)
    sin_eps = np.sqrt(1 - cos_eps**2)
    mu = l * (
        2 * m**2 * (R2**2 - R3**2 - l**2) + alpha_sq * (R4**2 - R3**2 - m**2)
    ) / (4 * l**2 * m**2 - alpha_sq**2)
    nu = m * (
        2 * l**2 * (R4**2 - R3**2 - m**2) + alpha_sq * (R2**2 - R3**2 - l**2)
    ) / (4 * l**2 * m**2 - alpha_sq**2)
    d_sq = np.abs(R3**2  - mu**2 - nu**2 + 2 * mu * nu * cos_eps)
    d = np.sqrt(d_sq)

    # calculate mutual inductance
    Omega = (
        np.arctan((d_sq * cos_eps + (mu + l) * (nu + m) * sin_eps**2) / (d * R1 * sin_eps))
        - np.arctan((d_sq * cos_eps + (mu + l) * nu * sin_eps**2) / (d * R2 * sin_eps))
        + np.arctan((d_sq * cos_eps + mu * nu * sin_eps**2) / (d * R3 * sin_eps))
        - np.arctan((d_sq * cos_eps + mu * (nu + m) * sin_eps**2) / (d * R4 * sin_eps))
    )
    M = 0.001 * cos_eps * ( 2 * (
        (mu + l) * np.arctanh(m/(R1 + R2)) + (nu + m) * np.arctanh(l / (R1 + R4))
        - mu * np.arctanh(m/ (R3 + R4)) - nu * np.arctanh(l / (R2 + R3))
    ) - (Omega * d / sin_eps))
    return M * sign * 10**-6 # convert to H and account for sign

# function that puts it all together
def calc_partial_mutual_ind(points1, points2) : 
    threshold = 1e-4 # threshold value for rounding errors
    # change to centimeters
    points1 = points1 * 100
    points2 = points2 * 100
    # make A, B, C, D
    C = points1[:, 0]
    D = points1[:, 1]
    A = points2[:, 0]
    B = points2[:, 1]
    AB = B - A # A to B / B - A
    CD = D - C # C to D / D - C

    # check cases to determine which equation to use
    # first check if perpendicular
    perp_bool  = np.abs(np.dot(AB, CD) / (np.linalg.norm(AB) * np.linalg.norm(CD))) < threshold 
    # check for parallel with cross product
    cross_bool = np.linalg.norm(np.cross(AB, CD)) < threshold
    # check for coplanar by volume of parallelpiped
    AC = C - A
    coplanar_bool = np.abs(np.dot(AC, np.cross(AB, CD))) < threshold

    # run through cases
    if perp_bool :
        M = 0
    elif cross_bool : 
        M = parralel_lines(A, B, C, D)
    elif coplanar_bool and ~cross_bool : # if cross bool is true, coplanar bool will always be true
        M = coplanar_lines(A, B, C, D)
    else : # if its none of these, its any desired position case...
        M = any_desired_position(A, B, C, D)
    return M

# function to calculate mutual inductance from coil objects
def calc_M_pim(coil1, coil2, matrix=False) : 
    coords1 = coil1.coords
    coords2 = coil2.coords

    total_M = 0
    M_matrix = np.zeros((coords1.shape[1] - 1, coords2.shape[1] - 1))


    # iterate through coil
    for ind1 in range(coords1.shape[1] - 1) :
        # get points for first line
        points1 = coords1[:, ind1:ind1+2]
        for ind2 in range(coords2.shape[1] - 1) : 
            # get points for second line
            points2 = coords2[:, ind2:ind2+2]
            partial_M = calc_partial_mutual_ind(points1, points2)
            total_M += partial_M
            M_matrix[ind1, ind2] = partial_M

    if matrix :
        return total_M, M_matrix
    else :
        guy = 0
        guy = guy + 1
        return total_M

# Function to calculate Self inductance through PIM with coil object
def calc_L_pim(coil, matrix = False) : 
    coords1 = coil.coords

    total_self_L = 0
    L_matrix = np.zeros((coords1.shape[1] - 1, coords1.shape[1] - 1))


    # iterate through coil
    for ind1 in range(coords1.shape[1] - 1) :
        # get points for first line
        points1 = coords1[:, ind1:ind1+2]
        for ind2 in range(coords1.shape[1] - 1) : 
            # get points for second line
            points2 = coords1[:, ind2:ind2+2]

            if ind1 == ind2 : # calculate self inductance of wire
                # depending on cross section, calculate partial self inductance
                if coil.cross_sec == 'circular' :
                    partial_L = L_part_hubert(points1, coil.wire_radius)
                else :
                    partial_L = L_part_rect_cross(points1, coil.wire_height, coil.wire_width)
            else : # calculate partial mutual inductance
                partial_L = calc_partial_mutual_ind(points1, points2) # only here am I changing to cm
            
            # add to total self inductance and matrix
            L_matrix[ind1, ind2] = partial_L
            total_self_L += partial_L

    if matrix :
        return total_self_L, L_matrix
    else :
        return total_self_L

# function to calculate self inductance of a coil with cirular cross section
def alternate_self_inductance(coil):
    # Make a coil that is the same but with the radius and pitch added to the wire radius
    coil_alt = Coil(
        coil.x_shift, coil.y_shift, coil.z_shift, coil.init_rad + coil.wire_radius, 
        coil.a, coil.turns_per_layer, coil.seg_per_turn, coil.cross_sec, 
        coil.wire_radius, coil.wire_height, coil.wire_width, coil.layers, coil.layer_distance, 
        coil.theta_direction, coil.layer_direction, coil.rot_x, coil.rot_y, coil.rot_z
    )

    # Calculate mutual inductance
    L_alt = calc_M_pim(coil, coil_alt)
    return L_alt

# function to generate inductance matrix from a file path
def generate_ind_matrix(file_path, jupiter = False):
    # file_path is the path to the txt file with information
    # jupiter is a boolean that is true if using a jupiter notebook

    # start time
    start_time = time.time()

    # Ensure that the output folder exists before proceeding
    with open(file_path, 'r') as file:
        for line in file:
            if "output folder" in line.lower():
                output_folder = line.split(":")[1].strip()
                break
        else:
            raise ValueError("Output folder not specified in the file.")
            
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Generate the coils from the file
    coils = generate_coils_from_txt(file_path)

    # Check to see if any are circular cross-section
    is_circular = np.array([coil.cross_sec == 'circular' for coil in coils])
    # If any are circular, make an extra matrix for the alternate calculation
    if np.any(is_circular):
        M_matrix_alt = np.zeros((len(coils), len(coils)))

    # Calculate the mutual inductance matrix
    M_matrix = np.zeros((len(coils), len(coils)))  # Create an empty matrix
    for i in range(len(coils)):
        for j in range(len(coils)):
            if i == j:
                M_matrix[i, j] = calc_L_pim(coils[i])
                break  # Avoid double calculations for self-inductance
            else:
                M_ind = calc_M_pim(coils[i], coils[j])
                M_matrix[i, j] = M_ind
                M_matrix[j, i] = M_ind

    # Calculate alternate method for self-inductance if any are circular
    if np.any(is_circular):
        # Make matrix to store results
        M_matrix_alt = M_matrix.copy()
        # Index through is_circular and calculate self-inductance of circular coils
        for ind, is_circ in enumerate(is_circular):
            if is_circ:
                M_matrix_alt[ind, ind] = alternate_self_inductance(coils[ind])

    # Save results to a txt file
    file_name = file_path.split("/")[-1]
    output_file = output_folder + file_name.split(".")[0] + "_inductance_matrix.txt"
    # Write to the file
    with open(output_file, 'w') as f:
        # Write the first description
        f.write("Mutual Inductance Matrix\n")
        f.write("File: " + file_name + "\n\n")
        f.write("Format:\n [L11, M12, M13, ...]\n [M21, L22, M23, ...]\n [M31, M32, L33, ...]\n ...\n\n")

        # Save the inductance matrix
        f.write("Inductance Matrix:\n")
        np.savetxt(f, M_matrix, fmt='%e', delimiter=', ')

        # Save the alternate method results if any are circular
        if np.any(is_circular):
            # Write the second description
            f.write("\n\nAlternate Method for Self Inductance for Circular Cross Sections\n")
            np.savetxt(f, M_matrix_alt, fmt='%e', delimiter=', ')

        # end time
        end_time = time.time()
        # Write the time taken to the file
        f.write(f"\n\nTime taken: {end_time - start_time:.2f} seconds")

    # Make images of the coils
    plot_coils(coils, output_folder + "all_coils.jpg")
    for ind, coil in enumerate(coils) :
        plot_coils([coil], output_folder + f"coil_{ind}.jpg")

    
    if jupiter : # if using a jupiter notebook, print on screen
        print("Mutual Inductance Matrix\n")
        print("File: " + file_name + "\n\n")
        print("Format:\n [L11, M12, M13, ...]\n [M21, L22, M23, ...]\n [M31, M32, L33, ...]\n ...\n\n")

        # Save the inductance matrix
        print("Inductance Matrix:\n")
        print(np.array2string(M_matrix, formatter={'float_kind': lambda x: f'{x:.6e}'}, separator=', '))

        # Save the alternate method results if any are circular
        if np.any(is_circular):
            # Write the second description
            print("\n\nAlternate Method for Self Inductance for Circular Cross Sections\n")
            print(np.array2string(M_matrix_alt, formatter={'float_kind': lambda x: f'{x:.6e}'}, separator=', '))

        # print time
        print(f"\n\nTime taken: {end_time - start_time:.2f} seconds")

        # print coils to screen
        plot_coils(coils)
        for ind, coil in enumerate(coils) :
            plot_coils([coil])

