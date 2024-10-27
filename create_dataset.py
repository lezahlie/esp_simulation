from setup_logger import setup_logger
logger = setup_logger(__file__, log_stdout=True, log_stderr=True)
from arguments import process_args
from utilities import *
from electrostatic_simulation import generate_electrostatic_maps
from plot_samples import plot_simulation_samples

# processes function gets shape maps and saves to a file in chunks
def process_image_maps(data_file, seed_range, seed_step, 
                        grid_length, material_cell_ratio, conductive_material_ratio, 
                        enable_fixed_charges, enable_absolute_permittivity, 
                        max_iterations, convergence_tolerance, 
                        plot_path, simvp_format, 
                        shared_data, shared_lock):

    cp_pid = current_process().pid
    remove_if_exists(data_file)

    for sr in split_seed_range(seed_range, seed_step):
        sim_results = generate_electrostatic_maps(min_seed=sr[0], 
                                                    max_seed=sr[1], 
                                                    grid_length=grid_length, 
                                                    material_cell_ratio=material_cell_ratio, 
                                                    conductive_material_ratio=conductive_material_ratio, 
                                                    enable_fixed_charges=enable_fixed_charges, 
                                                    enable_absolute_permittivity=enable_absolute_permittivity, 
                                                    max_iterations=max_iterations,
                                                    convergence_tolerance=convergence_tolerance,
                                                    images_only=simvp_format)
        # save the data chunk to the hdf5 file
        save_to_hdf5(sim_results, data_file, seed_step)

        # update the global min and max for all scalers and images
        local_min_max = compute_min_max_results(sim_results, simvp_format)
        update_shared_data(local_min_max, shared_data, shared_lock)

        # save 1 sample plot if enabled
        if plot_path is not None:
            sample_dicts = read_from_hdf5(data_file, sample_size=1)
            plot_simulation_samples(sample_dicts, plot_path, enable_fixed_charges, simvp_format)


# combines all results files into one big file
def gather_task_results(task_data_paths, final_file, seed_chunk):
    cp_pid = current_process().pid
    logger.info(f"PID[{cp_pid}]: Combining results files from each task into one file")
    remove_if_exists(final_file)
    combine_hdf5_files(task_data_paths, final_file, seed_chunk)
    logger.info(f"PID[{cp_pid}]: Saved combined shape maps to: {final_file}")


# creates and runs each process
def run_processes(task_data_paths, seed_range_per_task, default_args, simvp_format:bool=False):
    # shared data is to track global min and max for normalizing data 
    # this is useful for normalizing now or in the dataloader later
    manager = Manager()
    shared_data = manager.dict() 
    shared_lock = Lock()

    procs_list = []
    for i, (seed_range, task_file) in enumerate(zip(seed_range_per_task, task_data_paths)):
        p_args = [task_file, seed_range] + default_args + [shared_data, shared_lock]  # Pass shared dict to each process
        p = Process(target=process_image_maps, name=f"esp_simulation_p{i}", args=p_args)
        procs_list.append(p)
        p.start()
        logger.info(f"PID[{p.pid}]: child started")

    for p in procs_list:
        p.join()
        logger.info(f"PID[{p.pid}]: child joined")

    return dict(shared_data)


def main():
    args = process_args(__file__)

    # all the possible args
    req_cores = args.num_tasks
    min_seed = args.min_seed
    max_seed = args.max_seed
    seed_step = args.seed_step
    image_size = args.image_size
    normalize = args.normalize
    simvp_format = args.simvp_format
    convergence_tolerance = args.convergence_tolerance
    enable_fixed_charges = args.enable_fixed_charges
    enable_absolute_permittivity = args.enable_absolute_permittivity
    material_cell_ratio = args.material_cell_ratio
    conductive_material_ratio = args.conductive_material_ratio
    max_iterations = args.max_iterations
    total_seeds = (max_seed-min_seed+1)

    # friendly names
    solver_name = 'laplace' if enable_fixed_charges else 'poisson'
    datatype_name = "electrostatic"

    output_folder_path = path.join(args.output_path, args.output_folder)
    data_path = create_folder(f"{output_folder_path}")

    # creates output folder, file name prefix, file ext
    file_prefix = f"{datatype_name}_{solver_name}_{image_size}x{image_size}"
    file_fmt = "hdf5"
    
    # for saving plots
    plot_folder, plot_path = None, None
    if args.plot_samples:
        plot_folder = create_folder(f"{output_folder_path}/plots")
        plot_path = f"{plot_folder}/{file_prefix}" 

    # split up shapes between tasks(cores)
    seed_range_per_task = split_seed_range((min_seed, max_seed), total_seeds// req_cores)
    task_data_paths = [f"{data_path}/{file_prefix}_{seed_range[0]}-{seed_range[1]}.{file_fmt}" for seed_range in seed_range_per_task]

    # start process tasks
    logger.info(f"PID[{current_process().pid}]: parent process")
    
    default_args = [
                    seed_step, 
                    image_size, 
                    material_cell_ratio, 
                    conductive_material_ratio, 
                    enable_fixed_charges, 
                    enable_absolute_permittivity, 
                    max_iterations,
                    convergence_tolerance,
                    plot_path,
                    simvp_format
                ]
    
    global_min_max = run_processes(task_data_paths, seed_range_per_task, default_args)

    # combine process results
    if req_cores > 1:
        final_file_path = f"{data_path}/{file_prefix}_{min_seed}-{max_seed}.{file_fmt}"
        gather_task_results(task_data_paths, final_file_path , seed_step)
    else:
        final_file_path = task_data_paths[0]

    if simvp_format:
        # save to simvp formatted numpy files
        channels_minmax = normalize_hdf5_to_numpy(final_file_path, global_min_max, output_folder_path, datatype_name, normalize=normalize, chunk_size=seed_step)
        save_to_json(path.join(output_folder_path, "global_min_max_values_simvp.json"), channels_minmax)
        remove_if_exists(final_file_path)
    else:
        if normalize:
            # save normalized copy fo hdf5 file
            final_file_name = path.basename(final_file_path)
            new_final_file_name = path.join(path.dirname(final_file_path), f"normalized_{final_file_name}")
            normalize_hdf5_to_hdf5(final_file_path, new_final_file_name, global_min_max, chunk_size=seed_step)
            remove_if_exists(final_file_path)
        save_to_json(path.join(output_folder_path, "global_min_max_values_hdf5.json"), global_min_max)


if __name__ == "__main__":
    main()
