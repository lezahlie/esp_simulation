from setup_logger import setup_logger
logger = setup_logger(__file__, log_stdout=True, log_stderr=True)
from utilities import np, ndimg 
from electrostatic_mappers import epsilon_0, generate_initial_potential_map, neumann_boundary_conditions, dirichlet_boundary_conditions

# Poisson's equation using the Gauss–Seidel method
# https://en.wikipedia.org/wiki/Discrete_Poisson_equation
# https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method
def solve_poisson_equation(potential_map, charge_distribution, permittivity_map, 
                        max_iterations=1000, convergence_tolerance=1e-6, save_states=False):
    iteration = 0
    converged = False
    inverse_permittivity_map = np.where(permittivity_map != 0, 1.0 / permittivity_map, 0.0)
    intermediate_states = []

    while not converged and iteration < max_iterations:
        max_delta = 0.0

        if save_states and iteration > 0:
            intermediate_states.append(potential_map.copy())
                                    
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

    return potential_map, intermediate_states, (max_delta, converged, iteration) 

# Follows Laplace's equation using the Gauss–Seidel method
# https://en.wikipedia.org/wiki/Laplace%27s_equation
# https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method
def solve_laplace_equation(potential_map, max_iterations=1000, convergence_tolerance=1e-6, save_states=False):
    iteration = 0
    converged = False
    intermediate_states = []

    while not converged and iteration < max_iterations:
        max_delta = 0.0 
            
        if save_states and iteration > 0:
            intermediate_states.append(potential_map.copy())

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

    return potential_map, intermediate_states, (max_delta, converged, iteration) 

# Computes charge distribution from inverse of Poisson's equation
# by applying the Laplace operator to the potential map 
# https://en.wikipedia.org/wiki/Green%27s_function_for_the_three-variable_Laplace_equation
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

# Follows gauss's law for relative values: https://en.wikipedia.org/wiki/Gauss%27s_law
def generate_free_charge_distribution(conductive_material_mask, permittivity_map, kernel_size=5, expand_radius=1):
    # apply 4-neighborhood around conductive cells
    expanded_mask = ndimg.binary_dilation(conductive_material_mask, iterations=expand_radius)

    # generate random potential map with kernel smoothing
    smooth_potential = generate_smooth_potential(conductive_material_mask.shape, kernel_size)

    # expand the potential to prevent spikes
    synthetic_potential = smooth_potential * expanded_mask

    # calculate electric field gradients E(-P)
    E_x, E_y = np.gradient(-synthetic_potential)

    # calculate electric displacement D(x,y)
    D_x = permittivity_map * E_x
    D_y = permittivity_map * E_y

    # calculate the divergence of D(x,y) 
    div_D_x = np.gradient(D_x, axis=0)
    div_D_y = np.gradient(D_y, axis=1)

    # compute absolute free charge density 
    rho_free = div_D_x + div_D_y
    rho_free -= np.mean(rho_free)

    # restrict free charges to conductive regions
    free_charge_distribution = np.where(conductive_material_mask, rho_free, 0)

    return free_charge_distribution


def compute_electrostatic_potential(conductive_material_mask, permittivity_map, max_iterations, tolerance_value, voltage_range: tuple[int, int] = None, save_states:bool=False):

    input_potential_map = generate_initial_potential_map(conductive_material_mask, voltage_range)
    initial_potential = input_potential_map.copy()


    if voltage_range is None:
        charge_distribution = generate_free_charge_distribution(conductive_material_mask, permittivity_map)

        final_potential, intermediate_potential, (max_delta, completed, total_iterations) = solve_poisson_equation(input_potential_map,
                                                                                                                    charge_distribution, 
                                                                                                                    permittivity_map, 
                                                                                                                    max_iterations=max_iterations, 
                                                                                                                    convergence_tolerance=tolerance_value,
                                                                                                                    save_states=save_states) 

    else:
        final_potential, intermediate_potential, (max_delta, completed, total_iterations) = solve_laplace_equation(input_potential_map,
                                                                                                                    max_iterations=max_iterations, 
                                                                                                                    convergence_tolerance=tolerance_value,
                                                                                                                    save_states=save_states)
        charge_distribution = laplacian_operator_charge_distribution(final_potential, permittivity_map)

    solver_images= {
        "charge_distribution": charge_distribution,
        "initial_potential_map": initial_potential, 
        "final_potential_map": final_potential
    }

    if save_states and intermediate_potential:
        solver_images["intermediate_potential_states"] = np.stack(intermediate_potential)

    solver_meta = {
        "max_delta": max_delta,
        "converged" : int(completed),
        "total_iterations": total_iterations
    }

    return solver_images, solver_meta