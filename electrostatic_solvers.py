from setup_logger import setup_logger
logger = setup_logger(__file__, log_stdout=True, log_stderr=True)
from utilities import np, ndimg 
from electrostatic_mappers import generate_initial_potential_map, neumann_boundary_conditions, dirichlet_boundary_conditions


def solve_poisson_equation(potential_map, charge_distribution, permittivity_map, max_iterations=1000, convergence_tolerance=1e-6):
    iteration = 0
    converged = False
    inverse_permittivity_map = np.where(permittivity_map != 0, 1.0 / permittivity_map, 0.0)

    while not converged and iteration < max_iterations:
        max_delta = 0.0

        for i in range(1, charge_distribution.shape[0] - 1):
            for j in range(1, charge_distribution.shape[1] - 1):

                cell_update = ((charge_distribution[i, j] * inverse_permittivity_map[i, j]) +
                            ((potential_map[i-1, j] + potential_map[i+1, j] +
                                potential_map[i, j-1] + potential_map[i, j+1]) / 4))

                max_delta = max(max_delta, abs(potential_map[i, j] - cell_update))
                potential_map[i, j] = cell_update

        converged = max_delta < convergence_tolerance
        iteration += 1

    logger.debug(f"Iteration[{iteration}]: converged = {converged}, max delta = {max_delta}")
    assert np.any(potential_map != 0), "Poisson Solver: Potential map is all zeros!"

    return potential_map, (max_delta, converged, iteration)


def solve_laplace_equation(potential_map, max_iterations=1000, convergence_tolerance=1e-6):
    iteration = 0
    converged = False

    while not converged and iteration < max_iterations:
        max_delta = 0.0 
        for i in range(1, potential_map.shape[0] - 1):
            for j in range(1, potential_map.shape[1] - 1):

                cell_update = ((potential_map[i-1, j] + potential_map[i+1, j] +
                                potential_map[i, j-1] + potential_map[i, j+1]) / 4)
                
                max_delta = max(max_delta, abs(potential_map[i, j] - cell_update))
                potential_map[i, j] = cell_update

        potential_map = neumann_boundary_conditions(potential_map)
        converged = max_delta < convergence_tolerance
        iteration += 1

    logger.debug(f"Iteration[{iteration}]: converged = {converged}, max delta = {max_delta}")
    assert np.any(potential_map != 0), "Laplace Solver: Potential map is all zeros!"
    
    return potential_map, (max_delta, converged, iteration)


def laplacian_operator_charge_distribution(potential_map, permittivity_map):
    laplacian = np.zeros_like(potential_map)
    for i in range(1, potential_map.shape[0] - 1):
        for j in range(1, potential_map.shape[1] - 1):
            laplacian[i, j] = permittivity_map[i, j] * (potential_map[i-1, j] + potential_map[i+1, j] +
                                potential_map[i, j-1] + potential_map[i, j+1]) - (4 * potential_map[i, j])
    return laplacian


def generate_smooth_potential(shape, kernel_size=5):
    random_noise = np.random.rand(*shape)
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
    smoothed_potential = ndimg.convolve(random_noise, kernel, mode='reflect')
    return smoothed_potential


def generate_free_charge_distribution(conductive_material_mask, permittivity_map, kernel_size=5, expand_radius=1):
    # apply 4-neighborhood around conductive cells
    expanded_mask = ndimg.binary_dilation(conductive_material_mask, iterations=expand_radius)

    # generate random potential map with kernel smoothing
    smooth_potential = generate_smooth_potential(conductive_material_mask.shape, kernel_size)

    synthetic_potential = smooth_potential* expanded_mask

    # calculate electric field gradients along the X and Y
    grad_x, grad_y = np.gradient(synthetic_potential)

    # get the magnitude of the electric field
    electric_field_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # charge distribution is created from the electric field and permittivity
    charge_distribution = electric_field_magnitude * permittivity_map

    # restrict free charges to conductive regions
    free_charge_distribution = np.where(conductive_material_mask, charge_distribution, 0)
    return free_charge_distribution


def compute_electrostatic_potential(conductive_material_mask, permittivity_map, max_iterations, tolerance_value, voltage_range: tuple[int, int] = None):

    input_potential_map = generate_initial_potential_map(conductive_material_mask, voltage_range)
    initial_potential_map = input_potential_map.copy()

    if voltage_range is None:
        charge_distribution = generate_free_charge_distribution(conductive_material_mask, permittivity_map)

        final_potential_map, (max_delta, steady_state, total_iterations) = solve_poisson_equation(input_potential_map,
                                                                                                    charge_distribution, 
                                                                                                    permittivity_map, 
                                                                                                    max_iterations=max_iterations, 
                                                                                                    convergence_tolerance=tolerance_value) 

    else:
        final_potential_map, (max_delta, steady_state, total_iterations) = solve_laplace_equation(input_potential_map,
                                                                                                    max_iterations=max_iterations, 
                                                                                                    convergence_tolerance=tolerance_value)
        
        charge_distribution = laplacian_operator_charge_distribution(final_potential_map, permittivity_map)

    return (initial_potential_map, charge_distribution, final_potential_map), (max_delta, steady_state, total_iterations)
