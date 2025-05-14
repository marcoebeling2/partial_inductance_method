import numpy as np
import os
import time
import macros_classes as macro
from IPython.display import clear_output
import matplotlib
#matplotlib.use('Agg')  # Prevents Jupyter from auto-displaying plots
import matplotlib.pyplot as plt

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
def L_part_rect_cross(points, w, h) :
    # points are the coordinates of the wire (can calculate length from this)
    # w is cross sectional width
    # h is cross sectional height
    mu_o = 4 * np.pi * 10**-7
    l = np.linalg.norm(points[:, 1] - points[:, 0])
    D = 0.22313*(w+h)
    L = (mu_o / (2 * np.pi)) * l * (np.log((2 * l) / D) - 1)
    return L

# helper function
# assumes lengths in cm
def equal_parallel(l, d) : 
    M = 0.002 * l * (
        np.log((l/d) + np.sqrt(1 + (l/d)**2)) - np.sqrt(1 + (d/l)**2) + d/l
    )
    return M * 10**-6 # convert to H

def parralel_lines(A, B, C, D) :
    threshold = 1e-3 # threshold value for rounding errors
    # first check if the lines are parallel or anti parrallel
    AB = B - A
    CD = D - C
    l = np.linalg.norm(AB)
    m = np.linalg.norm(CD)
    multiplier = 1 # this is to have the appropriate sign for the mutual inductance
    # true if anti-parallel
    if ((np.dot(CD, AB) / (l * m)) + 1) < threshold :
        # switch the C and D
        C, D = D, C
        CD = D - C
        # change the multiplier to -1
        multiplier = -1
    

    # Seven further cases (6 cases with subcases)
    # 1. Exact overlap: the lines are the same length and both ends overlap each other
    # 2. Straight line: the lines are straight and share a common end
    # 3. No overlap: the lines share no overlap in their common plane
    # 4. Overlap: the lines overlap in the middle, but not completely overlap
    # 5. Total Engulfment: one line is completely inside the other
    # 6. Same End: the coils overlap and share a common end (seperated by distance d)
    # 7. Opposite End: the coils do not overlap but share a common end (seperated by distance d)

    # 1. Exact overlap: occurs when same length and first angle is perpendicular
    CA = A - C # from C to A... the convention I am using
    DCA_angle = np.arccos(np.dot(CA, CD) / (np.linalg.norm(CA) * m)) # angle between CA and CD
    DC = C - D
    DA = A - D
    CDA_angle = np.arccos(np.dot(DC, DA) / (np.linalg.norm(DC) * np.linalg.norm(DA))) # angle between DC and DA
    exact_overlap = (np.abs(l - m) < threshold) and (np.abs(np.pi/2 - DCA_angle) < threshold)

    # 2. Straight line: AC is parallel to AB
    AC = C - A
    straight_line = (np.linalg.norm(np.cross(AC, AB)) / (np.linalg.norm(AC) * np.linalg.norm(AB))) < threshold

    # 3. No overlap: occurs when CBA or CDA are obtuse
    BC = C - B
    BA = A - B
    CBA_angle = np.arccos(np.dot(BC, BA) / (np.linalg.norm(BC) * np.linalg.norm(BA))) # angle between BC and BA
    no_overlap = ((CDA_angle - np.pi/2) > threshold) or ((CBA_angle - np.pi/2) > threshold)

    # 4. Overlap: Occurs when case_1:(CAB is accute and DBA is obtuse) or case_2:(DCA is acute and CDB is obtuse)
    AC = C - A
    AB = B - A
    CAB_angle = np.arccos(np.dot(AB, AC) / (np.linalg.norm(AB) * np.linalg.norm(AC))) # angle between AB and AC
    BA = A - B
    BD = D - B
    DBA_angle = np.arccos(np.dot(BD, BA) / (np.linalg.norm(BD) * np.linalg.norm(BA))) # angle between BD and BA
    case_1 = ((np.pi/2 - CAB_angle) > threshold) and ((DBA_angle - np.pi/2) > threshold)
    DB = B - D
    CDB_angle = np.arccos(np.dot(DB, DC) / (np.linalg.norm(DB) * np.linalg.norm(DC))) # angle between DB and DC
    case_2 = ((np.pi/2 - DCA_angle) > threshold) and ((CDB_angle - np.pi/2) > threshold)
    overlap = case_1 or case_2

    # 5. Total Engulfment: occurs when DCA and CDB are acute or when CAB and DBA are acute
    both_DCA_CDB = ((np.pi/2 - DCA_angle) > threshold) and ((np.pi/2 - CDB_angle) > threshold)
    both_CAB_DBA = ((np.pi/2 - CAB_angle) > threshold) and ((np.pi/2 - DBA_angle) > threshold)
    total_engulfment = both_DCA_CDB or both_CAB_DBA

    # 6. Same End: Occurs if CA is perpendicular to CD or if DB is perpendicular CD, but not both. Handle this after exact overlap, so can just test for perpendicularity regardless of length of lines
    same_end = (np.abs(DCA_angle - np.pi/2) < threshold) or (np.abs(CDB_angle - np.pi/2) < threshold)

    # 7. Opposite End: Occurs when CBA or DAB are perpendicular
    opposite_end = (np.abs(CBA_angle - np.pi/2) < threshold) or (np.abs(CDA_angle - np.pi/2) < threshold)

    # calculate distance between lines
    d = np.linalg.norm(AC) * np.sin(CAB_angle)


    # go through the cases and apply the proper formulas from Grover 1973
    if exact_overlap :
        M = equal_parallel(l, d)
        return equal_parallel(l, d) * multiplier#, "exact_overlap"
    elif straight_line : # eq 35 and 36
        # check if the ends are touching: either B and C are the same or D and B are the same
        touch = (np.linalg.norm(B - C) < threshold) or  (np.linalg.norm(D - A) < threshold)
        if touch : 
            M = 0.001 * (
                l * np.log((l + m) / l) + m * np.log((l + m) / m)
            ) # mutual inductance calculation
            return M * multiplier * 10**-6 # convert to H
        else :
            # calculate distance between the two
            delta = min(np.linalg.norm(BC), np.linalg.norm(DA))
            M = 0.001 * (
                (l + m + delta) * np.log(l + m + delta)
                - (l + delta) * np.log(l + delta)
                - (m + delta) * np.log(m + delta)
                + delta * np.log(delta)
            )
            return M * multiplier * 10**-6#, "straight_line" # convert to H
    elif no_overlap :
        # calculate parallel distance between the two
        # if CDA angle is obtuse...
        if (CDA_angle - np.pi/2) > 0 :
            delta = np.linalg.norm(DA) * np.sin(CDA_angle - np.pi/2)
        else : # if CBA angle is obtuse
            delta = np.linalg.norm(BC) * np.sin(CBA_angle - np.pi/2)
        M = 0.5 * (
            (equal_parallel(l + m + delta,d) + equal_parallel(delta, d)) - (equal_parallel(l + delta, d) + equal_parallel(m + delta, d))
        )
        return M * multiplier#, "no_overlap"
    elif overlap :
        # calculate the parallel overlap distance
        delta = min(np.sqrt(np.linalg.norm(DA)**2 - d**2), np.sqrt(np.linalg.norm(BC)**2 - d**2))
        M = 0.5 * (
            (equal_parallel(l + m - delta,d) + equal_parallel(delta, d)) - (equal_parallel(l - delta, d) + equal_parallel(m - delta, d))
        )
        return M * multiplier#, "overlap"

    elif total_engulfment : # page 46 grover 1973
        # calculate distances between the ends
        p = np.sqrt(np.linalg.norm(CA)**2 - d**2)
        q = np.sqrt(np.linalg.norm(DB)**2 - d**2)
        M = 0.5 * (
            (equal_parallel(min(m, l) + p, d) + equal_parallel(min(m, l) + q, d)) - (equal_parallel(p, d) + equal_parallel(q, d))
        )
        return M * multiplier#, "total_engulfment"
    elif same_end :
        M = 0.5 * (
            equal_parallel(l, d) + equal_parallel(m, d) - equal_parallel(max(m, l) - min (m, l), d)
        )
        return M * multiplier#, "same_end"
    elif opposite_end :
        M = 0.5 * (
            equal_parallel(l + m, d) - equal_parallel(l, d) - equal_parallel(m, d)
        )
        return M * multiplier#, "opposite_end"
    else :
        print("Error: No case applied. This should not happen.")
        print("A: ", A, "B: ", B, "\nC: ", C, "D: ", D)
        print("AB: ", AB, "CD: ", CD)

        raise ValueError("NaN in parallel calculation")
    

# rect cross section via grover equal parallel
# assumes lengths in cm
def self_L_rect_grover(l, w, h) :
    # this is from table 3 in grover 1973
    ratios = np.linspace(0, 1, 21)
    outputs = np.array([0, 0.00146, 0.00210, 0.00239, 0.00249, 0.00249, 0.00244,
                        0.00236, 0.00228, 0.00219, 0.00211, 0.00203,
                        0.00197, 0.00192, 0.00187, 0.00184, 0.00181, 0.00179,
                        0.00178, 0.00177, 0.00177])
    
    # interpolate the value
    ratio = min(w/h, h/w)
    value = np.interp(ratio, ratios, outputs)

    # calculate the mutual inductance
    L = 0.002*l* (
        np.log((2*l)/(w+h)) + 0.5 - np.log(value)
    )
    return L * 10**-6 # convert to H


### coplanar cases
def coplanar_lines(A, B, C, D) : 
    threshold = 1e-1 # threshold value for rounding errors
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
    threshold = 1e-3 # threshold value for rounding errors
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
        #case_ = "perp"
    elif cross_bool : 
        M = parralel_lines(A, B, C, D)
    elif coplanar_bool and ~cross_bool : # if cross bool is true, coplanar bool will always be true
        M = coplanar_lines(A, B, C, D)
        #case_ = "coplanar"
    else : # if its none of these, its any desired position case...
        M = any_desired_position(A, B, C, D)
        #case_ = "ADP"
    return M#, case_

# function to calculate mutual inductance from coil objects
def calc_M_pim(coil1, coil2, matrix=False) : 
    coords1 = coil1.coords
    coords2 = coil2.coords

    total_M = 0
    M_matrix = np.zeros((coords1.shape[1] - 1, coords2.shape[1] - 1))
    # holds cases
    #cases_matrix = np.empty((coords1.shape[1] - 1, coords2.shape[1] - 1), dtype=object)


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
            #cases_matrix[ind1, ind2] = case_

    if matrix :
        return total_M, M_matrix#, cases_matrix
    else :
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
                # call the coil's method for calculating partial self inductance
                partial_L = coil.calc_L_partial(points1)
            else : # calculate partial mutual inductance
                partial_L = calc_partial_mutual_ind(points1, points2) # only here am I changing to cm
            
            # add to total self inductance and matrix
            L_matrix[ind1, ind2] = partial_L
            total_self_L += partial_L

    # save calculations within the coil
    coil.self_inductance_matrix = L_matrix
    coil.self_inductance = total_self_L

    if matrix :
        return total_self_L, L_matrix
    else :
        return total_self_L

# function to calculate self inductance of a coil with cirular cross section
def alternate_self_inductance(coil):
    # Make a coil that is the same but with the radius and pitch added to the wire radius
    coil_alt = macro.Coil(
        coil.x_shift, coil.y_shift, coil.z_shift, coil.init_rad + coil.wire_radius, 
        coil.a, coil.turns_per_layer, coil.seg_per_turn, coil.cross_sec, 
        coil.wire_radius, coil.wire_height, coil.wire_width, coil.layers, coil.layer_distance, 
        coil.theta_direction, coil.layer_direction, coil.rot_x, coil.rot_y, coil.rot_z
    )

    # Calculate mutual inductance
    L_alt = calc_M_pim(coil, coil_alt)
    return L_alt

# function to generate inductance matrix from a file path
def run_computations(file_path, jupiter = False, export = False):
    # file_path is the path to the txt file with information
    # jupiter is a boolean that is true if using a jupiter notebook
    # export is a boolean if you want to use the matrix for other calculations

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
    coils = macro.parse_coil_file(file_path)

    # Calculate the mutual inductance matrix
    M_matrix = calc_M_matrix(coils)

    # stop timer
    end_time = time.time()

    # caclulate resistances
    resistances = generate_res(coils)

    # compute Q factors
    Q_factors = calc_Q(coils, M_matrix, resistances)

    

    # Save results to a txt file
    file_name = file_path.split("/")[-1]
    output_file = output_folder + file_name.split(".")[0] + "_pim_summary.txt"
    # Write to the file
    with open(output_file, 'w') as f:
        # Write the first description
        f.write("Mutual Inductance Matrix\n")
        f.write("File: " + file_name + "\n\n")
        f.write("Format:\n [L11, M12, M13, ...]\n [M21, L22, M23, ...]\n [M31, M32, L33, ...]\n ...\n\n")

        # Save the inductance matrix
        f.write("Inductance Matrix (uH):\n")
        np.savetxt(f, M_matrix*10**6, fmt='%7.3f', delimiter=', ')

        # Save the resistances
        f.write("\n\nResistances (mOhms):\n")
        np.savetxt(f, resistances*10**3, fmt='%7.3f', delimiter=', ')

        # Save the Q factors
        f.write("\n\nQ Factors:\n")
        np.savetxt(f, np.diagonal(Q_factors), fmt='%10.3f', delimiter=', ')

        # Write the time taken to the file
        f.write(f"\n\nTime taken: {end_time - start_time:.2f} seconds")

    # save L matrix to a csv file
    output_file_csv = output_folder + file_name.split(".")[0] + "_inductance.csv"
    # Write to the file
    with open(output_file_csv, 'w') as f:
        # Save the inductance matrix
        np.savetxt(f, M_matrix, fmt='%.12f', delimiter=', ')

    # save R matrix to a csv file
    output_file_csv = output_folder + file_name.split(".")[0] + "_resistance.csv"
    # Write to the file
    with open(output_file_csv, 'w') as f:
        # Save the inductance matrix
        np.savetxt(f, resistances, fmt='%.6f', delimiter=', ')

    # Make images of the coils
    plt.ioff() # turn off interactive plotting
    macro.plot_coils(coils, file_path=output_folder + "all_coils.jpg")
    for ind, coil in enumerate(coils) :
        macro.plot_coils([coil], file_path=output_folder + f"coil_{ind}.jpg")
    plt.ion() # turn on interactive plotting

    
    if jupiter : # if using a jupiter notebook, print on screen
        #clear_output(wait=True) # clear previous output
        print("Mutual Inductance Matrix\n")
        print("File: " + file_name + "\n\n")
        print("Format:\n [L11, M12, M13, ...]\n [M21, L22, M23, ...]\n [M31, M32, L33, ...]\n ...\n\n")

        # Save the inductance matrix
        print("Inductance Matrix (uH):\n")
        print(np.array2string(M_matrix, formatter={'float_kind': lambda x: f'{(x*10**6):7.3f}'}, separator=', '))

        # Save the resistances
        print("\n\nResistances (mOhms):\n")
        print(np.array2string(resistances, formatter={'float_kind': lambda x: f'{(x*10**3):7.3f}'}, separator=', '))

        # Save the Q factors
        print("\n\nQ Factors:\n")
        print(np.array2string(np.diagonal(Q_factors), formatter={'float_kind': lambda x: f'{x:7.3f}'}, separator=', '))
        
        # print time
        print(f"\n\nTime taken: {end_time - start_time:.2f} seconds")

        # print coils to screen
        macro.plot_coils(coils)
        for ind, coil in enumerate(coils) :
            macro.plot_coils([coil])
    
    if export :
        return M_matrix, resistances, Q_factors, coils

# function to calculate the M matrix of a system of coils
def calc_M_matrix(coils) :
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
    return M_matrix

# function to generate resistance of each coil
def generate_res(coils) :
    resistances = np.zeros(len(coils))
    # calculate resistance of each coil
    for ind, coil in enumerate(coils) :
        coil.calc_resistance()
        resistances[ind] = coil.resistance
    return resistances

# function to calculate the Q factor of a coil
def calc_Q(coils, M_matrix, resistances) :
     # compute Q factors
    Q_factors = np.zeros((len(resistances), len(resistances)))  # Create an empty matrix
    # calculate diagnoals first
    for i in range(len(resistances)):
        Q_factors[i, i] = 2 * np.pi * coils[i].frequency * M_matrix[i, i] / resistances[i] # Q = omega * L / R
    for i in range(len(resistances)):
        for j in range(len(resistances)):
            if i == j:
                break  # Avoid double calculations for self-inductance
            else:
                k = M_matrix[i, j] / np.sqrt(M_matrix[i, i] * M_matrix[j, j])
                Q_factors[i, j] = Q_factors[i, i] * Q_factors[j, j] / (k * (Q_factors[i, i] + Q_factors[j, j]))
                Q_factors[j, i] = Q_factors[i, j]

    return Q_factors

