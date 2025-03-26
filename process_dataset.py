from setup_logger import setup_logger, set_logger_level
logger = setup_logger(__file__, log_stdout=True, log_stderr=True)
from arguments import process_args
from utilities import glob, path, DATATYPE_NAME, DEFAULT_DATAFILE_EXT, SIMVP_DATAFILE_EXT, current_process, create_folder, read_from_hdf5, read_from_json, save_to_json, normalize_hdf5_to_hdf5, normalize_hdf5_to_numpy
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

    global_extrema_file = None
    global_extrema_values = None
    new_dataset_file_path = None

    # find the global extreme values file
    extrema_file_paths = glob(f"{dataset_folder_path}/global_extrema_{DEFAULT_DATAFILE_EXT}_{dataset_file_name.replace(DEFAULT_DATAFILE_EXT, 'json')}")
    if len(extrema_file_paths) != 1:
        raise FileNotFoundError(f"Cannot find global extrema json file in path: {dataset_folder_path}")
    global_extrema_file = extrema_file_paths[0]
    global_extrema_values = read_from_json(extrema_file_paths[0])
    global_extrema_postfix = dataset_file_name.replace(DEFAULT_DATAFILE_EXT, 'json')

    if simvp_format:
        # save to simvp formatted numpy files
        channels_minmax_values = normalize_hdf5_to_numpy(dataset_file_path, global_extrema_values, output_folder_path, DATATYPE_NAME, normalize=normalize)
        new_global_extrema_file = f"global_extrema_{SIMVP_DATAFILE_EXT}_{global_extrema_postfix}"
        save_to_json(path.join(output_folder_path, new_global_extrema_file), channels_minmax_values)
    elif normalize:
        # save normalized copy to hdf5 file
        new_dataset_file_path = path.join(output_folder_path, f"normalized_{dataset_file_name}")
        normalize_hdf5_to_hdf5(dataset_file_path, new_dataset_file_path, global_extrema_values)
        if output_folder_path != dataset_folder_path:
            save_to_json(path.join(output_folder_path, path.basename(global_extrema_file)), global_extrema_values)

    # save sample plots
    if sample_plots == 0:
        exit(0)

    if new_dataset_file_path:
        plot_dataset_file = new_dataset_file_path
        global_extrema_values = {key: (0.0, 1.0) for key in global_extrema_values['image'].keys()}
    else:
        plot_dataset_file = dataset_file_path
        global_extrema_values = global_extrema_values['image']

    plot_prefix = "_".join(path.basename(plot_dataset_file).split('_')[:-1])
    plot_path = create_folder(path.join(output_folder_path, "plots"))
    sample_dicts = read_from_hdf5(plot_dataset_file, sample_size=sample_plots)
    plot_simulation_samples(sample_dicts, plot_path, plot_prefix, global_extrema_values, plot_states=plot_states)



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(e)
