from setup_logger import setup_logger
logger = setup_logger(__file__, log_stdout=True, log_stderr=True)
from arguments import process_args
from utilities import path, current_process, create_folder, read_from_hdf5, read_from_json, save_to_json, normalize_hdf5_to_hdf5, normalize_hdf5_to_numpy
from plot_samples import plot_simulation_samples


def main():
    args = process_args(__file__)

    # all the possible args
    normalize = args.normalize
    simvp_format = args.simvp_format
    dataset_file_path = args.dataset_path
    sample_plots = args.sample_plots

    dataset_folder_path = path.abspath(path.dirname(dataset_file_path))
    dataset_file_name = path.basename(dataset_file_path)

    output_folder_path = (path.join(args.output_path, args.output_folder) 
                            if args.output_folder is not None 
                            else dataset_folder_path)

    create_folder(output_folder_path)

    # start process tasks
    logger.info(f"PID[{current_process().pid}]: parent process")
    datatype_name = "electrostatic"

    global_minmax_file_path = f"{dataset_folder_path}/global_extrema_hdf5_{dataset_file_name.replace('hdf5', 'json')}"
    global_minmax_values  = read_from_json(global_minmax_file_path)

    if simvp_format:
        # save to simvp formatted numpy files
        channels_minmax_values = normalize_hdf5_to_numpy(dataset_file_path, global_minmax_values, output_folder_path, datatype_name, normalize=normalize)
        new_global_minmax_file = f"global_extrema_simvp_{dataset_file_name.replace('hdf5', 'json')}"
        save_to_json(path.join(output_folder_path, new_global_minmax_file), channels_minmax_values)
    elif normalize:
        # save normalized copy fo hdf5 file
        new_dataset_file_name = path.join(output_folder_path, f"normalized_{dataset_file_name}")
        normalize_hdf5_to_hdf5(dataset_file_path, new_dataset_file_name, global_minmax_values)
        if output_folder_path != dataset_folder_path:
            save_to_json(path.join(output_folder_path, path.basename(global_minmax_file_path)), global_minmax_values)

    # save sample plots
    plot_path = None
    if sample_plots > 0:
        plot_prefix = "_".join(dataset_file_name.split('_')[:-1])
        plot_path = create_folder(path.join(output_folder_path, "plots"))
        sample_dicts = read_from_hdf5(dataset_file_path, sample_size=sample_plots)
        plot_simulation_samples(sample_dicts, global_minmax_values, f"{plot_path}/{plot_prefix}", simvp_format)



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(e)
