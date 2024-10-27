from setup_logger import setup_logger
logger = setup_logger(__file__, log_stdout=True, log_stderr=True)
from utilities import np, create_folder, read_from_hdf5, save_to_hdf5
from electrostatic_mappers import *
from electrostatic_metrics import *
from electrostatic_solvers import compute_electrostatic_potential
from plot_samples import plot_simulation_samples

def run_electrostatic_simulation(simulation_config: dict, seed: int = None, images_only:bool=False):
    if seed:
        np.random.seed(seed)

    grid_length = simulation_config.get('grid_length', 32)
    material_ratio = simulation_config.get('material_cell_ratio', 1.0)
    conductive_ratio = simulation_config.get('conductive_material_ratio', 0.25)
    voltage_range = simulation_config.get('voltage_range', None) 
    max_iterations = simulation_config.get('max_iterations', 2000)
    convergence_tolerance = simulation_config.get('convergence_tolerance', 1e-6)

    # material ternary mask of categories: free_space, insulated, conductors
    material_category_mask, conductive_material_mask = create_material_location_masks(grid_length, material_ratio, conductive_ratio)

    # maps random material index to cells based on category mask
    material_index_map = generate_material_index_map(material_category_mask)
    
    mask_dict = {
        "material_category_map": material_category_mask,
        "conductive_material_map": conductive_material_mask,
        "material_id_map": material_index_map
    }

    # permittivity map of all material permittivity values
    permittivity_value_map = generate_permittivity_value_map(material_index_map)

    # solve for charge_distribution and final_potential_map
    (initial_potential_map, charge_distribution, final_potential_map), (max_delta, completed, total_iterations) = compute_electrostatic_potential(conductive_material_mask,
                                                                                                                                                    permittivity_value_map, 
                                                                                                                                                    max_iterations, 
                                                                                                                                                    convergence_tolerance,
                                                                                                                                                    voltage_range)
    meta_dict = {
        "random_seed": seed, 
        "image_size": grid_length,
        "max_delta":max_delta,
        "converged": int(completed),
        "total_iterations":total_iterations
    }

    image_dict = {
        "initial_potential_map": initial_potential_map,
        "permittivity_map": permittivity_value_map,
        "charge_distribution": charge_distribution,
        "final_potential_map":  final_potential_map,
    }

    simulation_result = {
        'meta': meta_dict,
        "mask": mask_dict,
        "image": image_dict
    }

    if images_only:
        return simulation_result

    # compute electric field across x and y
    electric_field_x, electric_field_y = compute_electric_field(final_potential_map)
    # compute magnitude of the electric field x and y
    electric_field_magnitude = np.sqrt(electric_field_x**2 + electric_field_y**2)

    image_dict["electric_field_x"] = electric_field_x
    image_dict["electric_field_y"] = electric_field_y
    image_dict["electric_field_magnitude"] = electric_field_magnitude


    # compute electric flux across boundaries
    electric_flux = compute_electric_flux(electric_field_x, electric_field_y)

    # compute total energy in the electric_field x and y
    total_energy = compute_total_energy(electric_field_x, electric_field_y, permittivity_value_map)

    # compute total charge density from charge_distribution
    total_charge = np.sum(charge_distribution)

    metric_dict = {
        "electric_flux": electric_flux,
        "total_energy": total_energy,
        "total_charge": total_charge
    }

    # @note unused for now
    #potential_diff_x, potential_diff_y = compute_local_potential_differences(final_potential_map)
    #global_potential_diff = compute_global_potential_differences(final_potential_map, reference_point=(grid_length // 2, grid_length // 2))
    #pairwise_potential_diff = compute_pairwise_potential_differences(final_potential_map)

    simulation_result["image"].update(image_dict)
    simulation_result["metric"] = metric_dict

    return simulation_result


def generate_electrostatic_maps(min_seed: int=0, 
                                max_seed: int=100, 
                                grid_length=32, 
                                material_cell_ratio=1.0,
                                conductive_material_ratio=0.25, 
                                enable_fixed_charges=False, 
                                enable_absolute_permittivity=False,
                                max_iterations=2000,
                                convergence_tolerance=1e-6,
                                images_only=False):


    set_permittivity_type(enable_absolute_permittivity)
    
    simulation_config = {
        'grid_length': grid_length,  
        'conductive_material_ratio': conductive_material_ratio,
        'material_cell_ratio': material_cell_ratio,
        'max_iterations': max_iterations,
        'voltage_range': (50, 200) if enable_fixed_charges else None,
        'convergence_tolerance': convergence_tolerance,
        'enable_fixed_charges': enable_fixed_charges
    }

    simulation_data = []
    for seed in range(min_seed, max_seed + 1):
        result_dict = run_electrostatic_simulation(simulation_config, seed=seed, images_only=images_only)
        simulation_data.append(result_dict)
        #logger.debug("Simulation Result:\n%s", pprint.pformat(result_dict))
    return simulation_data


def main():
    grid_length = 32
    min_seed = 1
    max_seed = 5

    conductive_material_ratio = 0.5
    material_cell_ratio=1.0

    enable_absolute_perm=False
    max_iterations=5000
    convergence_tolerance=1e-6

    # testing fixed charges
    data_folder = create_folder(f"simulation_temp/data")
    data_file = f"dataset_{grid_length}x{grid_length}_{min_seed}-{max_seed}.hdf5"
    plot_folder = create_folder(f"simulation_temp/plots")

    fixed_charges=True
    result_dataset = generate_electrostatic_maps(min_seed=min_seed, 
                                                max_seed=max_seed, 
                                                grid_length=grid_length, 
                                                material_cell_ratio=material_cell_ratio,
                                                conductive_material_ratio=conductive_material_ratio,
                                                enable_fixed_charges=fixed_charges, 
                                                enable_absolute_permittivity=enable_absolute_perm,
                                                max_iterations=max_iterations,
                                                convergence_tolerance=convergence_tolerance,
                                                images_only=fixed_charges)
    
    laplace_data_path = f"{data_folder}/laplace_{data_file}"
    save_to_hdf5(result_dataset, laplace_data_path)

    # testing fixed charges
    sample_dicts = read_from_hdf5(laplace_data_path, sample_size=5)
    plot_path_prefix = f"{plot_folder}/laplace_images_{grid_length}x{grid_length}"
    plot_simulation_samples(sample_dicts, plot_path_prefix, fixed_charges, minimal=False)

    fixed_charges=False
    result_dataset = generate_electrostatic_maps(min_seed=min_seed, 
                                                max_seed=max_seed, 
                                                grid_length=grid_length, 
                                                material_cell_ratio=material_cell_ratio,
                                                conductive_material_ratio=conductive_material_ratio,
                                                enable_fixed_charges=fixed_charges, 
                                                enable_absolute_permittivity=enable_absolute_perm,
                                                max_iterations=max_iterations,
                                                convergence_tolerance=convergence_tolerance,
                                                images_only=fixed_charges)

    poisson_data_path = f"{data_folder}/poisson_{data_file}"
    save_to_hdf5(result_dataset, poisson_data_path)

    sample_dicts = read_from_hdf5(poisson_data_path, sample_size=5)
    plot_path_prefix = f"{plot_folder}/poisson_images_{grid_length}x{grid_length}"
    plot_simulation_samples(sample_dicts, plot_path_prefix, fixed_charges, minimal=False)


if __name__ == "__main__":
    main()