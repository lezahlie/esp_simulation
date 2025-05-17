from setup_logger import setup_logger, set_logger_level
logger = setup_logger(__file__, log_stdout=True, log_stderr=True)
from arguments import process_args
from utilities import (glob, 
                        path, 
                        DATATYPE_NAME,
                        DEFAULT_DATAFILE_EXT, 
                        SIMVP_DATAFILE_EXT, 
                        create_folder, 
                        read_from_hdf5, 
                        read_from_json, 
                        save_to_json, 
                        normalize_hdf5_to_hdf5, 
                        normalize_hdf5_to_numpy,
                        extract_minmax_tuples)
from plot_samples import plot_simulation_samples


def main():
    args = process_args(__file__)

    if args.debug_on:
        set_logger_level(10)

    # all the possible args
    normalize = args.normalize
    simvp_format = args.simvp_format
    dataset_file_path = args.dataset_path
    sample_plots = args.sample_plots
    plot_states = args.plot_states
    dataset_folder_path = path.abspath(path.dirname(dataset_file_path))
    dataset_file_name = path.basename(dataset_file_path)

    output_folder_path = (path.join(args.output_path, args.output_folder) 
                            if args.output_folder is not None 
                            else dataset_folder_path)

    create_folder(output_folder_path)

    global_statistics_file = None
    global_statistics_values = None
    new_dataset_file_path = None

    # find the global extreme values file
    global_statistics_file = f"{dataset_folder_path}/global_statistics_{DEFAULT_DATAFILE_EXT}_{dataset_file_name.replace('normalized_', '').replace(DEFAULT_DATAFILE_EXT, 'json')}"
    statistics_file_paths = glob(global_statistics_file)
    if len(statistics_file_paths) != 1:
        raise FileNotFoundError(f"Cannot find global statistics json file in path: {global_statistics_file}")
    global_statistics_file = statistics_file_paths[0]
    global_statistics_values = read_from_json(statistics_file_paths[0])
    global_statistics_postfix = dataset_file_name.replace(DEFAULT_DATAFILE_EXT, 'json')

 
    if simvp_format:
        # save to simvp formatted numpy files
        channel_statistics = normalize_hdf5_to_numpy(dataset_file_path, global_statistics_values, output_folder_path, DATATYPE_NAME, normalize=normalize)
        new_global_statistics_file = f"global_statistics_{SIMVP_DATAFILE_EXT}_{global_statistics_postfix}"
        save_to_json(path.join(output_folder_path, new_global_statistics_file), channel_statistics)
    elif normalize:
        # save normalized copy to hdf5 file
        new_dataset_file_path = path.join(output_folder_path, f"normalized_{dataset_file_name}")
        normalize_hdf5_to_hdf5(dataset_file_path, new_dataset_file_path, global_statistics_values)
        if output_folder_path != dataset_folder_path:
            save_to_json(path.join(output_folder_path, path.basename(global_statistics_file)), global_statistics_values)
    elif 'normalized' in dataset_file_name:
        new_dataset_file_path = dataset_file_path

    # save sample plots
    if sample_plots == 0:
        exit(0)

    if new_dataset_file_path:
        plot_dataset_file = new_dataset_file_path
        global_minmax_tuples = {key: [0.0, 1.0] for key in global_statistics_values['image'].keys()}
    else:
        plot_dataset_file = dataset_file_path
        global_minmax_tuples = extract_minmax_tuples(global_statistics_values)

    plot_prefix = "_".join(path.basename(plot_dataset_file).split('_')[:-1])
    plot_path = create_folder(path.join(output_folder_path, "plots"))
    sample_dicts = read_from_hdf5(plot_dataset_file, sample_size=sample_plots)
    plot_simulation_samples(sample_dicts, plot_path, plot_prefix, global_minmax_tuples, plot_states=plot_states)



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(e)
