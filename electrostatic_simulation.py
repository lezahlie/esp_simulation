from setup_logger import setup_logger
logger = setup_logger(__file__, log_stdout=True, log_stderr=True)
from utilities import np, create_folder, read_from_hdf5, save_to_hdf5
from electrostatic_mappers import *
from electrostatic_metrics import *
from electrostatic_solvers import compute_electrostatic_potential
from connect_materials import connect_cellular_automata_shape

def run_electrostatic_simulation(simulation_config: dict, seed: int = None, images_only:bool=False, save_states:bool=False):
    if seed:
        np.random.seed(seed)

    conductive_cell_ratio = simulation_config.get('conductive_cell_ratio', None)
    conductive_cell_prob = simulation_config.get('conductive_cell_prob', None)
    conductive_material_count = simulation_config.get('conductive_material_count', None)
    conductive_material_range = simulation_config.get('conductive_material_range', None)
    grid_length = simulation_config.get('grid_length', 32)
    max_iterations = simulation_config.get('max_iterations', 2000)
    convergence_tolerance = simulation_config.get('convergence_tolerance', 1e-6)
    voltage_range = simulation_config.get('fixed_voltage_range', None) 

    mask_dict = {}
    # material ternary mask of categories: free_space, insulated, conductors
    material_category_mask, conductive_material_mask = create_material_category_mask(grid_length, 
                                                                                    conductive_cell_ratio=conductive_cell_ratio, 
                                                                                    conductive_cell_prob=conductive_cell_prob)

    # apply CA algo to conductive cells and connect disjoint regions
    conn_conductive_material_mask = connect_cellular_automata_shape(conductive_material_mask, reverse=True)
    mask_dict["conductive_material_map"] = conn_conductive_material_mask

    # update the material catagory map with the new conductive shape
    conn_material_category_mask = update_material_category_mask(material_category_mask, conn_conductive_material_mask)
    mask_dict["material_category_map"] = conn_material_category_mask

    # map all the materials based on the new mask
    conn_material_index_map = generate_material_index_map(conn_material_category_mask, 
                                                            conductive_material_range=conductive_material_range,
                                                            conductive_material_count=conductive_material_count)
    mask_dict["material_id_map"] = conn_material_index_map

    # permittivity map of all material permittivity values
    permittivity_value_map = generate_permittivity_value_map(conn_material_index_map)

    # solve for charge_distribution and potential_states 
    solver_images, solver_meta = compute_electrostatic_potential(conductive_material_mask,
                                                                permittivity_value_map, 
                                                                max_iterations, 
                                                                convergence_tolerance,
                                                                voltage_range, 
                                                                save_states)

    meta_dict = {
        "random_seed": seed, 
        "image_size": grid_length,
        **solver_meta
    }

    image_dict = {
        "permittivity_map": permittivity_value_map,
        **solver_images
    }

    simulation_result = {
        'meta': meta_dict,
        "mask": mask_dict,
        "image": image_dict
    }

    if images_only:
        return simulation_result

    # compute electric field across x and y
    electric_field_x, electric_field_y = compute_electric_field(image_dict['final_potential_map'])

    # @note unused for now to save compute time

    # compute electric flux across boundaries
    electric_flux = compute_electric_flux(electric_field_x, electric_field_y)

    # compute total energy in the electric_field x and y
    total_energy = compute_total_energy(electric_field_x, electric_field_y, permittivity_value_map)

    # compute total charge density from charge_distribution
    total_charge = np.sum(image_dict['charge_distribution'])

    metric_dict = {
        "electric_flux": electric_flux,
        "total_energy": total_energy,
        "total_charge": total_charge
    }

    # @note unused for now to save compute time
    # electric_field_magnitude = np.sqrt(electric_field_x**2 + electric_field_y**2)
    # potential_diff_x, potential_diff_y = compute_local_potential_differences(final_potential_map)
    # global_potential_diff = compute_global_potential_differences(final_potential_map, reference_point=(grid_length // 2, grid_length // 2))
    # pairwise_potential_diff = compute_pairwise_potential_differences(final_potential_map)

    simulation_result["image"].update(image_dict)
    simulation_result["metric"] = metric_dict

    return simulation_result


def generate_electrostatic_maps(min_seed: int=0, 
                                max_seed: int=100, 
                                grid_length:int=32, 
                                conductive_cell_ratio:float|None=None, 
                                conductive_cell_prob:float|None=None, 
                                conductive_material_count:int|None=None,
                                conductive_material_range:tuple[int,int]|None=None,
                                enable_fixed_charges=False, 
                                enable_absolute_permittivity=False,
                                max_iterations=2000,
                                convergence_tolerance=1e-6,
                                images_only=False, 
                                save_states=False):


    set_permittivity_type(enable_absolute_permittivity)

    if conductive_cell_ratio is None and conductive_cell_prob is None:
        raise ValueError("conductive_cell_prob and conductive_cell_ratio are both None")
    elif isinstance(conductive_cell_ratio, float) and isinstance(conductive_cell_prob, float):
        raise ValueError("conductive_cell_prob and conductive_cell_ratio are mutually exclusive")
    elif conductive_material_count is None and conductive_material_range is None:
        raise ValueError("conductive_material_count and conductive_material_range are both None")
    elif isinstance(conductive_material_count, int) and isinstance(conductive_material_range, tuple[int,int]):
        raise ValueError("conductive_material_count and conductive_material_range are mutually exclusive")
    
    simulation_config = {
        'grid_length': grid_length,  
        'conductive_cell_ratio': conductive_cell_ratio, 
        'conductive_cell_prob': conductive_cell_prob, 
        'conductive_material_count': conductive_material_count,
        'conductive_material_range':  tuple(conductive_material_range),
        'max_iterations': max_iterations,
        'convergence_tolerance': convergence_tolerance,
        'enable_fixed_charges': enable_fixed_charges,
        'fixed_voltage_range': (50, 200) if enable_fixed_charges else None
    }

    simulation_data = []
    for seed in range(min_seed, max_seed + 1):
        result_dict = run_electrostatic_simulation(simulation_config, seed=seed, images_only=images_only, save_states=save_states)
        simulation_data.append(result_dict)
        #logger.debug("Simulation Result:\n%s", pprint.pformat(result_dict))
    return simulation_data


def main():
    grid_length = 32
    min_seed = 1
    max_seed = 5

    enable_absolute_perm=False
    max_iterations=5000
    convergence_tolerance=1e-6

    # testing fixed charges
    data_folder = create_folder(f"simulation_temp/data")
    data_file = f"dataset_{grid_length}x{grid_length}_{min_seed}-{max_seed}.hdf5"
    plot_folder = create_folder(f"simulation_temp/plots")

    fixed_charges=True
    
    conductive_cell_ratio=None
    conductive_cell_prob=0.5
    conductive_material_count=None
    conductive_material_range=(1,6)

    result_dataset = generate_electrostatic_maps(min_seed=min_seed, 
                                                max_seed=max_seed, 
                                                grid_length=grid_length, 
                                                conductive_cell_ratio=conductive_cell_ratio, 
                                                conductive_cell_prob=conductive_cell_prob, 
                                                conductive_material_count=conductive_material_count,
                                                conductive_material_range=conductive_material_range,
                                                enable_fixed_charges=fixed_charges, 
                                                enable_absolute_permittivity=enable_absolute_perm,
                                                max_iterations=max_iterations,
                                                convergence_tolerance=convergence_tolerance,
                                                images_only=fixed_charges)
    
    laplace_data_path = f"{data_folder}/laplace_{data_file}"
    save_to_hdf5(result_dataset, laplace_data_path)

    fixed_charges=False
    result_dataset = generate_electrostatic_maps(min_seed=min_seed, 
                                                max_seed=max_seed, 
                                                grid_length=grid_length, 
                                                conductive_cell_ratio=conductive_cell_ratio, 
                                                conductive_cell_prob=conductive_cell_prob, 
                                                conductive_material_count=conductive_material_count,
                                                conductive_material_range=conductive_material_range,
                                                enable_fixed_charges=fixed_charges, 
                                                enable_absolute_permittivity=enable_absolute_perm,
                                                max_iterations=max_iterations,
                                                convergence_tolerance=convergence_tolerance,
                                                images_only=fixed_charges)

    poisson_data_path = f"{data_folder}/poisson_{data_file}"
    save_to_hdf5(result_dataset, poisson_data_path)


if __name__ == "__main__":
    main()