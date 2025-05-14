import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
import pim_methods as pim
import pandas as pd


# parent class for any coil
class Coil(ABC) :
    def __init__ (self, relative_permeability = 1, conductivity = 5.8e7, dielectric_constant = 4.3, frequency = 1e6, name = None) : 
        self.relative_permeability = relative_permeability # permeability of conducting material
        self.mu_o = 4 * np.pi * 10**-7
        self.mu = self.mu_o * relative_permeability
        self.conductivity = conductivity # conductivity of conducting material
        self.dielectric_constant = dielectric_constant # dielectric constant of the material (FR4 is around 4.3)
        self.frequency = frequency # default is 1 MHz (quasi-static)
        self.self_inductance = None # self inductance of the coil
        self.self_inductance_matrix = None # self inductance matrix of the coil
        self.resistance = None # resistance of the coil
        self.coords = None # coordinates of the coil

    # define abstract methods
    @abstractmethod
    def parse_lines(self, lines) :
        # abstract method that parses txt data to obtain coil parameters
        pass

    @abstractmethod
    def initialize(self, dict) :
        # abstracte method that initializes the coil parameters from the parsed data
        pass

    @abstractmethod
    def generate_coords(self) :
        # abstract method that generate the coordinates of the coil
        pass

    @abstractmethod
    def calc_L_partial(self, points) :
        # abstract method that calculate the self partial inductance for the coil
        # points are the coordinates of the wire segment to calculate the self inductance for
        pass

    def calc_resistance(self) :
        # calculate the resistance of the coil
        # does not have to be defined for all derived classes
        raise NotImplementedError(f"{self.__class__.__name__} does not support calc_resistance() method.")

# dervied class: racectangular spiral
class PlanarRectangularSpiral(Coil) :
    def __init__(self, param_dict = None, **kwargs) :
        # initialize parent class with default values. can change default values when read in data from txt
        super().__init__()

        # if calling directly (ex ipynb), 
        # pass **kwargs to initialize or dictionary to initialize
        if param_dict is not None :
            self.initialize(param_dict)
        elif kwargs: # executes is kwargs is not empty
            self.initialize(kwargs)

    def parse_lines(self, lines) :
        # Define a dictionary that maps line prefixes to variable assignments (make sure to convert to millimeters if applicable)
        param_df = pd.DataFrame([
            ("x-shift:", "x_shift", lambda v: float(v) / 1000),
            ("y-shift:", "y_shift", lambda v: float(v) / 1000),
            ("z-shift:", "z_shift", lambda v: float(v) / 1000),
            ("inner radius:", "i_radius", lambda v: float(v) / 1000 if float(v) > 0 else ValueError("init_rad must be positive.")),
            ("gap:", "gap", lambda v: float(v) / 1000 if float(v) > 0 else ValueError("gap must be positive.")),
            ("turns per layer:", "n_turns", lambda v: int(v) if int(v) > 0 else ValueError("turns_per_layer must be positive.")),
            ("layers:", "layers", lambda v: int(v) if int(v) > 0 else ValueError("layers must be positive.")),
            ("distance between layers:", "distance", lambda v: float(v) / 1000),
            ("cross sectional wire height:", "height", lambda v: float(v) / 1000 if float(v) > 0 else ValueError("wire_height must be positive.")),
            ("cross sectional wire width:", "width", lambda v: float(v) / 1000 if float(v) > 0 else ValueError("wire_width must be positive.")),
            ("rotation about x-axis:", "rot_x", lambda v: float(v)),
            ("rotation about y-axis:", "rot_y", lambda v: float(v)),
            ("rotation about z-axis/phase shift:", "rot_z", lambda v: float(v)),
            ("name:", "name", lambda v: v),
            ("frequency (mhz)", "frequency", lambda v: float(v) * 10**6),
            ("material relative conductivity", "conductivity", lambda v: float(v)*5.8*10**7),
            ("save info", "file_out", lambda v: v),
        ], columns=["key", "var_name", "processing_func"])
        parsed_values = {}  # Dictionary to store parsed values
        for line in lines:
            if line.lower().startswith("type:".lower()):
                 continue # skip the type line
            for _, row in param_df.iterrows():  # Iterate through DataFrame rows
                key, var_name, processing_func = row["key"], row["var_name"], row["processing_func"]

                if line.lower().startswith(key.lower()):  # Match key in line
                    try:
                        value = line.split(":", 1)[1].strip()  # Extract value after ':'
                        parsed_values[var_name] = processing_func(value)  # Apply function and store result
                        break  # Break out of the loop once a match is found

                    except (ValueError, TypeError) as e:
                        raise TypeError(f"Error processing '{key}': {e}")
        
        # pass the parsed_values to initialize
        self.initialize(parsed_values)

    def initialize(self, dict) :
        # check to see if any default arguments are None
        if dict.get("dielectric_constant") is None :
            dict["dielectric_constant"] = 4.3
        if dict.get("relative_permeability") is None :
            dict["relative_permeability"] = 1
        if dict.get("conductivity") is None :
            dict["conductivity"] = 5.8e7
        if dict.get("rot_x") is None :
            dict["rot_x"] = 0
        if dict.get("rot_y") is None :
            dict["rot_y"] = 0
        if dict.get("rot_z") is None :
            dict["rot_z"] = 0
        if dict.get("frequency") is None:
            dict["frequency"] = 1e6
        if dict.get("layers") is None :
            dict["layers"] = 1
        if dict.get("distance") is None :
            dict["distance"] = 0
        if dict.get("name") is None :
            dict["name"] = None

        # initialize values
        for key, value in dict.items():
            setattr(self, key, value)
        
        # generate the coordinates
        self.coords = self.generate_coords()

        # set the limit for plotting
        self.limit_for_plotting = 2 * ( self.i_radius + self.n_turns * (self.width + self.gap)) # limit for plotting
    
    # function to generate coordinates
    def generate_coords(self) :
        # generate coordinates for each layer and concatinate accordingly
        # iterate through layers
        for layer in range(self.layers) :
            # generate the coordinates for that layer
            layer_coords = self.make_planar_rectangular_layer()
            # flip direction if necessary
            if layer % 2 == 1 :
                # apply transformations
                # first rotate 180 degrees about the y
                rotation_matrix = np.array([
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, 0, -1]])
                layer_coords = rotation_matrix @ layer_coords
                # second rotate -90 degrees about the z axis
                rotation_matrix = np.array([
                    [0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 1]])
                layer_coords = rotation_matrix @ layer_coords
                # reverse the order of coordinates
                layer_coords = np.flip(layer_coords, axis=1)

            # make rotation matrices and apply if necessary
            # must apply x and y to the layer shift as well
            layer_shift = np.array([0, 0, layer * (self.distance + self.height)]).reshape(3,1) # shift in z direction
            if self.rot_x != 0:
                rotation_matrix = np.array([
                    [1, 0, 0],
                    [0, np.cos(self.rot_x), np.sin(self.rot_x)],
                    [0, -np.sin(self.rot_x), np.cos(self.rot_x)]])
                layer_coords = rotation_matrix @ layer_coords
                layer_shift = rotation_matrix @ layer_shift
            if self.rot_y != 0:
                rotation_matrix = np.array([
                    [np.cos(self.rot_y), 0, -np.sin(self.rot_y)],
                    [0, 1, 0],
                    [np.sin(self.rot_y), 0, np.cos(self.rot_y)]])
                layer_coords = rotation_matrix @ layer_coords
                layer_shift = rotation_matrix @ layer_shift
            if self.rot_z != 0:
                rotation_matrix = np.array([
                    [np.cos(self.rot_z), np.sin(self.rot_z), 0],
                    [-np.sin(self.rot_z), np.cos(self.rot_z), 0],
                    [0, 0, 1]])
                layer_coords = rotation_matrix @ layer_coords
                layer_shift = rotation_matrix @ layer_shift
            
            # shift the coordinates z-wise for each layer
            layer_coords += layer_shift


            # concentatinate here....
            # concatinate to the main coordinates
            if layer == 0 :
                coords = layer_coords
            else :
                coords = np.concatenate((coords, layer_coords), axis=1)
            
        # shift the coordinates (have to do this after rotations)
        coords[0, :] += self.x_shift
        coords[1, :] += self.y_shift
        coords[2, :] += self.z_shift

        return coords

    # helper function to generate coordinates for a single layer
    def make_planar_rectangular_layer(self) :
        # Make a planar rectangular coil
        # allocate space for the points
        n_points = self.n_turns * 4 + 1
        points = np.zeros((3, n_points))

        # overall equation is as follows:
        # l = (2r + w) + (g+w) * ((i - 3)/2) + k (k is for first and last one)
        # first point will start at origin
        # from there, the spiral goes counter clockwise (-X, -Y, +X, +Y directions)
        base = 2 * self.i_radius + self.width # base line length for each turn
        direction = np.array([[-1, 0, 0], [0, -1, 0], [1, 0, 0], [0, 1, 0]], dtype=float).T  # direction vector

        for ind in range(len(points.T) - 1):
            if ind == 0 : # first turn / length
                l = base - self.gap - float(self.width/2)
            elif ind < 3 :
                # if its still on the first turn, exclude the term about index (it will make it 0)
                l = base
            else : # other case
                l = base + (self.width + self.gap) * (1+((ind - 3) // 2))

                # handle end case
                if ind == len(points.T) - 2 :
                    l = l - self.gap - self.width/2
            
            # append the direction and lentgh to the last point and add to points array
            points[:, ind + 1] = points[:, ind] + l * direction[:, (ind) % 4]


        # center layer coords before returning
        points[0, :] += np.abs((points[0, 2] + points[0, 3]) / 2)
        points[1, :] += np.abs((points[1, 1] + points[1, 2]) / 2)        
        return points
    
    # function to calculate self partial inductance
    def calc_L_partial(self, points) :
        return pim.L_part_rect_cross(points, self.width, self.height)

    def calc_parasitic_cap(self) : # equation 5
        # calculate lg
        do = np.linalg.norm(self.coords[:, -2] - self.coords[:, -3]) + self.width
        lg = (4 * (do - self.width * self.n_turns) * (self.n_turns - 1) 
            - self.gap * self.n_turns * (self.n_turns + 1))
        
        # calculate parasitic capacitance
        alpha = 0.9 # from paper
        beta = 0.1 # from paper
        Cp = ((alpha * 1 + beta * self.dielectric_constant) # use 1 for relativie permitvity of air
            * (8.854e-12) * lg * (self.height / self.gap))
        
        self.Cp = Cp

    def calc_resistance(self) :
        # look into the source on this: https://www.emisoftware.com/calculator/resistance-rectangular-wire/  ... page 155
        # first calculate lc (total trace lenbth). The big question is whether or not to count w/2 for each segment... i think its ngeligible
        lc = np.sum(np.linalg.norm(self.coords[:, 1:] - self.coords[:, :-1], axis=0)) 
        Rdc = lc / (self.conductivity * self.height * self.width) 
        self.res_DC = Rdc
        # calculate skin effect cooeficient
        skin_depth = 1 / (np.sqrt(np.pi * self.mu * self.frequency * self.conductivity))
        self.skin_depth = skin_depth # save skin_depth
        k = self.height / (
            skin_depth * (1 - np.exp(-self.height / skin_depth) * (1 + self.height / self.width))
        )
        # calculate Rs
        Rs = k * Rdc
        self.res_skin_depth = Rs
        
        if (np.abs(2*skin_depth - self.width) > 0) or (np.abs(2*skin_depth - self.height) > 0) :
            self.res_skin_depth = Rs
        else :
            self.res_skin_depth = Rs

        # calculate Rp
        Rp = 0.1 * Rdc * (
            ((4 * np.pi * 10e-7 * (2 * np.pi * self.frequency) * self.width**2)) 
            / (3.1 * 0.14 * (self.gap + self.width))
        ) ** 2
        self.res_proximity = Rp

        # calculate Rd
        Rd = np.tan(skin_depth) / (2*np.pi*self.frequency)
        self.res_ohmic = Rd

        # add up for total
        self.resistance = self.res_skin_depth + self.res_proximity + self.res_ohmic # can also added resistance of any attached capacitors

# function that plots each coil in the list of coils
def plot_coils(coils, lim = None, file_path=None, legend = False) :
    # supress output for now
    plt.ioff()

    # add set axis length
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for ind, coil in enumerate(coils) :
        # Plot coil
        label_str = coil.name if coil.name is not None else f"Coil {ind + 1}" # generate a name for the coil
        # make the first one blue and second red
        if ind == 0 :
            ax.plot(coil.coords[0, :], coil.coords[1, :], coil.coords[2, :], label=label_str, color='blue')
        elif ind == 1 :
            ax.plot(coil.coords[0, :], coil.coords[1, :], coil.coords[2, :], label=label_str, color='red')
        else :
            ax.plot(coil.coords[0, :], coil.coords[1, :], coil.coords[2, :], label=label_str)
        
        # find the max value for the limit
        if ind == 0 :
            lim = coil.limit_for_plotting
        else :
            if  coil.limit_for_plotting > lim :
                lim =  coil.limit_for_plotting

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Coil(s) Visualization')
    if legend: # add legend if specified
        ax.legend()

    # set axis limits
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    

    if file_path:
        plt.savefig(file_path, format='jpg', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        
    # Clear the figure (optional) for ipynb
    plt.clf()
    plt.close(fig)

def parse_coil_file(file_path):
    coils = [] # empty list to store coils
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()

        # Find the line starting with "Number of coil(s):"
        num_coils = 0
        for line in lines: # should be the first line but read until can find
            if line.lower().startswith("number of coil".lower()):
                num_coils = int(line.split(":")[1].strip())
                break
        else:
            # If  don't find the line, raise an error or handle the case
            raise ValueError("Number of coil(s) not specified in the file.")

        # Split data into chunks using '--------------------------------------------' as the separator
        chunks = []
        chunk = []

        started = False # boolean to check if have started reading a coil data
        for line in lines:
            # start with the first seperator line
            if line.strip().startswith("-" * 10):  # Detect separator line (min 10 dashes)
                if not started : # if have not started yet, start reading in
                    started = True
                    continue
                if started : # if already collected, store it
                    chunks.append(chunk) # append the chunk to the list of all the chunks
                    chunk = []  # Reset for the next chunk

            elif not started : # if have not started yet, skip the line
                continue
            else : # if it is not a seperator line, append to chunk
                chunk.append(line.strip())  # Collect data line (without leading/trailing spaces)
        
        # iterate through chunks and create coils
        for ind, chunk in enumerate(chunks):
            # the first line will tell which type of coil it is
            # iterate through until find the line that starts with "type of coil:"
            coil_type = None
            for line in chunk:
                if line.lower().startswith("type:".lower()):
                    coil_type = line.split(":")[1].strip().lower()
                    break
            if coil_type is None:
                raise ValueError(f"Type of coil not specified for coil {ind + 1}.")
            # call factory function to create the correct type of coil
            coil = get_class_from_name(coil_type)
            # call the coils function to parse and initialize itself
            coil.parse_lines(chunk)
            # append the coil to the list of coils
            coils.append(coil)

    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except ValueError as e:
        print(f"Error parsing data: {e}")
    
    # raise error if number of coils does not match number of chunks (User made error inputting data)
    if len(coils) != num_coils:
        raise ValueError(f"Number of coils ({num_coils}) does not match number of coils parsed ({len(coils)}).")

    return coils

# Factory function to create the correct class of coil
def get_class_from_name(class_name):
    class_map = {
        "planar rectangular spiral": PlanarRectangularSpiral,
    }
    
    if class_name in class_map:
        obj = class_map[class_name]()  # Instantiate the correct class
        return obj
    else:
        raise ValueError(f"Unknown coil type: {class_name}")

