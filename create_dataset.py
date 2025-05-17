from setup_logger import setup_logger, set_logger_level
logger = setup_logger(__file__, log_stdout=True, log_stderr=True)
from arguments import process_args
from utilities import *
from electrostatic_simulation import generate_electrostatic_maps

# processes function gets shape maps and saves to a file in chunks
def process_image_maps(data_file, 
                        seed_range, 
                        seed_step, 
                        grid_length,
                        conductive_cell_ratio, 
                        conductive_cell_prob, 
                        conductive_material_count,
                        conductive_material_range,
                        enable_fixed_charges, 
                        enable_absolute_permittivity, 
                        max_iterations, 
                        convergence_tolerance, 
                        save_states,
                        shared_data, 
                        shared_lock):

    remove_if_exists(data_file)

    for sr in split_seed_range(seed_range, seed_step):
        sim_results = generate_electrostatic_maps(min_seed=sr[0], 
                                                    max_seed=sr[1], 
                                                    grid_length=grid_length, 
                                                    conductive_cell_ratio=conductive_cell_ratio, 
                                                    conductive_cell_prob=conductive_cell_prob, 
                                                    conductive_material_count=conductive_material_count,
                                                    conductive_material_range=conductive_material_range,
                                                    enable_fixed_charges=enable_fixed_charges, 
                                                    enable_absolute_permittivity=enable_absolute_permittivity, 
                                                    max_iterations=max_iterations,
                                                    convergence_tolerance=convergence_tolerance,
                                                    save_states=save_states)
        # save the data chunk to the hdf5 file
        save_to_hdf5(sim_results, data_file, seed_step)

        # update the global min and max for all scalers and images
        local_stats = compute_local_stats(sim_results)
        print(local_stats)
        update_shared_data(local_stats, shared_data, shared_lock)
        print(local_stats)

# combines all results files into one big file
def gather_task_results(task_data_paths, final_file, seed_chunk):
    cp_pid = current_process().pid
    logger.info(f"PID[{cp_pid}]: Combining results files from each task into one file")
    remove_if_exists(final_file)
    combine_hdf5_files(task_data_paths, final_file, seed_chunk)
    logger.info(f"PID[{cp_pid}]: Saved combined shape maps to: {final_file}")


# creates and runs each process
def run_processes(task_data_paths, seed_range_per_task, default_args):
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
        logger.info(f"PID[{p.pid}]: child started running simulations for seeds {seed_range}")

    for p in procs_list:
        p.join()
        logger.info(f"PID[{p.pid}]: child joined parent")

    return dict(shared_data)


def main():
    args = process_args(__file__)

    if args.debug_on:
        set_logger_level(10)

    # all the possible args
    req_cores = args.num_tasks
    min_seed = args.min_seed
    max_seed = args.max_seed
    seed_step = args.seed_step
    image_size = args.image_size
    max_iterations = args.max_iterations
    convergence_tolerance = args.convergence_tolerance
    enable_fixed_charges = args.enable_fixed_charges
    enable_absolute_permittivity = args.enable_absolute_permittivity
    save_states = args.save_states

    conductive_cell_ratio = getattr(args, 'conductive_cell_ratio', None)
    conductive_cell_prob = getattr(args, 'conductive_cell_prob', None)
    conductive_material_count = getattr(args, 'conductive_material_count', None)
    conductive_material_range = getattr(args, 'conductive_material_range', None)

    total_seeds = (max_seed-min_seed+1)
    # friendly names
    solver_name = 'laplace' if enable_fixed_charges else 'poisson'
    
    # creates output folder and data file prefix
    output_folder_path = path.join(args.output_path, args.output_folder)
    data_path = create_folder(output_folder_path)
    datafile_prefix = f"{DATATYPE_NAME}_{solver_name}_{image_size}x{image_size}"

    arguments_file_path = f"arguments_{datafile_prefix}_{min_seed}-{max_seed}.json"
    save_to_json(path.join(output_folder_path, arguments_file_path), vars(args))

    # split up shapes between tasks(cores)
    seed_range_per_task = split_seed_range((min_seed, max_seed), total_seeds// req_cores)
    task_data_paths = [f"{data_path}/{datafile_prefix}_{seed_range[0]}-{seed_range[1]}.{DEFAULT_DATAFILE_EXT}" for seed_range in seed_range_per_task]

    # start process tasks
    logger.info(f"PID[{current_process().pid}]: parent process")
    
    default_args = [
                    seed_step, 
                    image_size, 
                    conductive_cell_ratio, 
                    conductive_cell_prob, 
                    conductive_material_count,
                    conductive_material_range,
                    enable_fixed_charges, 
                    enable_absolute_permittivity, 
                    max_iterations,
                    convergence_tolerance,
                    save_states
                ]
    
    global_stats = run_processes(task_data_paths, seed_range_per_task, default_args)

    # combine process results
    if req_cores > 1:
        final_file_path = f"{data_path}/{datafile_prefix}_{min_seed}-{max_seed}.{DEFAULT_DATAFILE_EXT}"
        gather_task_results(task_data_paths, final_file_path , seed_step)
    else:
        final_file_path = task_data_paths[0]

    
    global_stats_file_name = path.basename(final_file_path).split('.')[0]
    global_stats_file_path = f"global_statistics_{DEFAULT_DATAFILE_EXT}_{global_stats_file_name}.json"
    save_to_json(path.join(output_folder_path, global_stats_file_path), global_stats)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(e)
