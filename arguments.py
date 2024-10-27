import argparse as ap
import utilities as util

executable_groups = {
    "electrostatic_simulation.py": ["image"],
    "create_dataset.py": ["data", "image", "multiprocess"]
}


def add_multiprocess_group(parser, file_name):
    group = parser.add_argument_group('multi-process options')
    group.add_argument('-t', '--ntasks', dest="num_tasks", type=int, default=1, 
                    help="Number of tasks (cpu cores) to run in parallel. If multi-threading is enabled, max threads is set to (num_tasks * 2) | default: 1")
    if "image" in executable_groups[file_name]:
        group.add_argument('-k', '--seed-step', dest="seed_step", type=int, default=50, 
                        help="Number of seeds to be processed and written at a time | default: 100")


def check_multiprocess_args(args):
    if not (1 <= args.num_tasks < util.cpu_count()):
        raise ValueError(f"NUM_TASKS must be a INT between [1, {util.cpu_count()} - 1]")


def add_image_group(parser):
    group = parser.add_argument_group("image generation options")
    group.add_argument('-s', '--image-size', dest='image_size', type=int, default=32, 
                    help="Length for one side of 2D image | default: 32")
    group.add_argument('-a', '--min-seed', dest='min_seed', type=int, default=1, 
                    help="Start seed for generating images from MIN_SEED to MAX_SEED | default: 1")
    group.add_argument('-b', '--max-seed', dest='max_seed', type=int, default=500, 
                    help="Ending seed for generating images from MIN_SEED to MAX_SEED | default: 100")
    group.add_argument('-m', '--material-cell-ratio', dest='material_cell_ratio', type=float, default=1.0,
                    help="Ratio of non border cells eligible for material placement | default: 1.0")
    group.add_argument('-c', '--conductive-material-ratio', dest='conductive_material_ratio', type=float, default=0.25,
                    help="Ratio of eligible material cells that should be conductive | default: 0.25")
    group.add_argument('-u', '--enable-absolute-permittivity', dest='enable_absolute_permittivity', action='store_true',
                    help="Enables converting material permittivity from relative to absolute | default: Off")
    group.add_argument('-l', '--enable-fixed-charges', dest='enable_fixed_charges', action='store_true',
                help="Enables solving for fixed charges, charges are free by default | default: Off")
    group.add_argument('-z', '--max-iterations', dest='max_iterations', type=int, default=2000,
                    help="Maximum allowed iterations to run electrostatic potential solvers | default: 2000")
    group.add_argument('-e', '--convergence-tolerance', dest='convergence_tolerance', type=float, default=1e-6,
                    help="Tolerance for convergence; reached when the maximum change between iterations falls below this value | default: 1E-6")

def check_image_args(args):
    if not (5 < args.image_size < 1025):
        raise ValueError("IMAGE_SIZE must be a INT between [5, 1024]")
    if not (0.5 <= args.material_cell_ratio <= 1.0):
        raise ValueError(f"MATERIAL_CELL_RATIO must be a FLOAT between (0.5, 1.0)")
    if not (0.2 < args.conductive_material_ratio <= 1.0):
        raise ValueError("CONDUCTIVE_MATERIAL_RATIO must be a FLOAT between (0.25, 1.0).")
    if not (1 <= args.min_seed < args.max_seed):
        raise ValueError("MIN_SEED must be a INT between [1, MAX_SEED - 1]")
    if not (args.min_seed < args.max_seed):
        raise ValueError("MAX_SEED must be a INT greater than MIN_SEED.")
    total_seeds = args.max_seed-args.min_seed+1
    if hasattr(args, 'seed_step') and not (0 < args.seed_step < total_seeds):
        raise ValueError(f"SEED_STEP must be a INT between [1, {total_seeds-1}]")
    if not (0 < args.max_iterations < 1e+9):
        raise ValueError("max_iterations must be a INT greater between [1, 1E9].")
    if not (0.0 <= args.convergence_tolerance <= 1.0):
        raise ValueError("CONVERGENCE_TOLERENCE must be a float between [0.0, 1.0].")

def add_data_group(parser,  file_name):
    group = parser.add_argument_group('data format options')
    group.add_argument('-o', '--output-path', dest='output_path', type=str, default='.',
                        help="Path the the directory to create [--output-folder] and save to | default: current directory")
    
    group.add_argument('-f', '--output-folder', dest='output_folder', default='espsim_dataset', type=str,
                        help="Output folder name to creave and save simulation data to | default: esp_dataset")
    
    group.add_argument('-w', '--disable-normalization', dest='normalize',  action='store_false', 
                    help="Option to disable normalizing simulation outputs with a min max scaler | default: Off")

    group.add_argument('-p', '--plot-samples', dest='plot_samples', action='store_true',
                    help="Option to save sample plots from electrostatic simulation | default: Off")

    group.add_argument('-v', '--simvp-format', dest='simvp_format', action='store_true',
                    help="Option to save dataset formatted for SimVP | default: Off")


def check_data_args(args):
    if hasattr(args, 'output_path') and not util.path.exists(args.output_path):
        raise FileNotFoundError(f"OUTPUT_PATH '{args.output_path}' does not exist")
    if not (0 < len(args.output_folder) < 129):
        raise ValueError(f"OUTPUT_FOLDER '{args.output_folder}' must have a length between [1, 128]")


def check_args(parser, file_name):
    args = parser.parse_args()
    if "data" in executable_groups[file_name]:
        check_data_args(args)
    if "image" in executable_groups[file_name]:
        check_image_args(args)
    if "multiprocess" in executable_groups[file_name]:
        check_multiprocess_args(args)
    return args


def process_args(exe_file):
    file_name = util.path.basename(exe_file)
    parser = ap.ArgumentParser(description="Electro Static Potential Simulation")
    # @note unused for now
    #parser.add_argument('-d', '--debug', dest='debug_on', action='store_true', 
                        #help="Enables debug option and verbose printing | default: off")
    if "data" in executable_groups[file_name]:
        add_data_group(parser, file_name)
    if "multiprocess" in executable_groups[file_name]:
        add_multiprocess_group(parser, file_name)
    if "image" in executable_groups[file_name]:
        add_image_group(parser)

    args = check_args(parser, file_name)
    return args