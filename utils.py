from scipy.ndimage import gaussian_filter, binary_erosion
from noise import snoise2
import numpy as np

def generate_normal_map(obj_mask):
    X_meshed, Y_meshed = generate_meshgrid(obj_mask.shape[0]) 
    # create contours 
    mask = obj_mask.astype(int)
    eroded_mask = binary_erosion(mask)
    contours = (mask - eroded_mask).astype(bool)
    
    # Apply a Gaussian filter to smooth the mask
    smoothed_mask = gaussian_filter(mask.astype(float), sigma=2)

    # Compute the gradient of the smoothed mask
    gradient_y_mask, gradient_x_mask = np.gradient(smoothed_mask)

    # Normalize to obtain unit vectors
    length_gradient = np.sqrt(gradient_x_mask**2 + gradient_y_mask**2)
    length_gradient[length_gradient == 0] = 1  # Avoid division by zero
    normal_y_mask = -gradient_y_mask / length_gradient  # Inversion for outward direction
    normal_x_mask = -gradient_x_mask / length_gradient  # Inversion for outward direction

    # Extract normal vectors only at detected contours
    normal_y_mask_contours = normal_y_mask[contours]
    normal_x_mask_contours = normal_x_mask[contours]
    Y_contours = Y_meshed[contours]
    X_contours = X_meshed[contours]
    return (Y_contours,
            X_contours,
            normal_y_mask_contours,
            normal_x_mask_contours)

def ellipse_generator(a, b):
    # size must be even, so divisible by 2.
    # size + 1 is odd, so the center have integers coordonates.
    # on top of that, linspaces with those constraint, generate integers.
    size = max(a,b)*2 + 1
    
    X_meshed, Y_meshed = generate_meshgrid(size)
    
    ellipse = ((X_meshed**2 / a**2 + Y_meshed**2 / b**2) <= 1)
    return ellipse

def generate_meshgrid(size):
    

    x = np.linspace(int(-size / 2), int(size / 2), size).astype(int)
    y = np.linspace(int(-size / 2), int(size / 2), size).astype(int)
    
    X_meshed, Y_meshed = np.meshgrid(x, y)
    return X_meshed, Y_meshed
    

def generate_linear_trajectory(start_point, end_point, N):
    start_point = np.array(start_point)
    end_point = np.array(end_point)
    
    # Générer les points
    trajectory = np.linspace(start_point, end_point, N)
    
    trajectory = np.floor(trajectory).astype(int) 
    
    return trajectory

def generate_nonflat_field(grid_size, random_seed):
    height_map = np.zeros(grid_size)
    
    for y in range(grid_size[0]):
        for x in range(grid_size[1]):
            new_value = snoise2(
                  x/grid_size[1],
                  y/grid_size[0],
                  octaves=3,
                  persistence=0.95,
                  lacunarity=2,
                  repeatx=grid_size[1],
                  repeaty=grid_size[0],
                  base=random_seed
                 )
            height_map[y][x] = int(50*new_value)
        
    return height_map
