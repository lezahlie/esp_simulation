from setup_logger import setup_logger
logger = setup_logger(__file__, log_stderr=True, log_stdout=True)
from utilities import np


def compute_electric_field(potential_map, distance=1):
    grad_x, grad_y = np.gradient(potential_map, distance)
    electric_field_x = -grad_x
    electric_field_y = -grad_y
    return electric_field_x, electric_field_y


def compute_electric_flux(electric_field_x, electric_field_y, area_element=1):
    flux_top = np.sum(electric_field_y[0, :]) * area_element        # Top boundary
    flux_bottom = np.sum(electric_field_y[-1, :]) * area_element    # Bottom boundary
    flux_left = np.sum(electric_field_x[:, 0]) * area_element       # Left boundary
    flux_right = np.sum(electric_field_x[:, -1]) * area_element     # Right boundary
    total_flux = flux_top + flux_bottom + flux_left + flux_right
    return total_flux


def compute_local_potential_differences(potential_map):
    potential_diff_x = np.diff(potential_map, axis=1)
    potential_diff_y = np.diff(potential_map, axis=0)
    return potential_diff_x, potential_diff_y


def compute_pairwise_potential_differences(potential_map):
    grid_size = potential_map.shape[0]
    diff_matrix = np.zeros((grid_size, grid_size, grid_size, grid_size))
    
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                for l in range(grid_size):
                    diff_matrix[i, j, k, l] = potential_map[i, j] - potential_map[k, l]
    return diff_matrix


def compute_global_potential_differences(potential_map, reference_point):
    ref_value = potential_map[reference_point[0], reference_point[1]]
    potential_diff = potential_map - ref_value
    return potential_diff


def compute_global_potential_differences(potential_map, reference_point):
    ref_value = potential_map[reference_point[0], reference_point[1]]
    potential_diff = potential_map - ref_value
    return potential_diff


def compute_total_energy(electric_field_x, electric_field_y, permittivity_map):
    energy_density = 0.5 * permittivity_map * (electric_field_x**2 + electric_field_y**2)
    total_energy = np.sum(energy_density)
    return total_energy