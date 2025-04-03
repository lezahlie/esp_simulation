import argparse as ap
import utilities as util
from electrostatic_mappers import conductive_indices


executable_groups = {
    "electrostatic_simulation.py": ["simulation", "output"],
    "create_dataset.py": ["simulation", "output", "multiprocess"],
    "process_dataset.py": ["dataset", "output"]
}


def parse_tuple(value):
    return tuple(map(int, value.strip("()").split(",")))


def add_multiprocess_group(parser, file_name):
    group = parser.add_argument_group('multi-process options')
    group.add_argument('-t', '--ntasks', dest="num_tasks", type=int, default=1, 
                    help="Number of tasks (cpu cores) to run in parallel. If multi-threading is enabled, max threads is set to (num_tasks * 2) | default: 1")
    if "simulation" in executable_groups[file_name]:
        group.add_argument('-k', '--seed-step', dest="seed_step", type=int, default=50, 
                        help="Number of seeds to be processed and written at a time | default: 100")


def check_multiprocess_args(args):
    if not (1 <= args.num_tasks < util.cpu_count()):
        raise ValueError(f"NUM_TASKS must be a INT between [1, {util.cpu_count()} - 1]")


def add_simulation_group(parser):
    group = parser.add_argument_group("image generation options")
    group.add_argument('-s', '--image-size', dest='image_size', type=int, default=32, 
                    help="Length for one side of 2D image | default: 32")
    group.add_argument('-a', '--min-seed', dest='min_seed', type=int, default=1, 
                    help="Start seed for generating images from MIN_SEED to MAX_SEED | default: 1")
    group.add_argument('-b', '--max-seed', dest='max_seed', type=int, default=500, 
                    help="Ending seed for generating images from MIN_SEED to MAX_SEED | default: 100")
    
    ccell_group = group.add_mutually_exclusive_group(required=True)
    ccell_group.add_argument('-r', '--conductive-cell-ratio', dest='conductive_cell_ratio', type=float,
                    help="Proportion of cells that should be conductive (Before cellular automata) | required")
    ccell_group.add_argument('-p', '--conductive-cell-prob', dest='conductive_cell_ratio', type=float,
                    help="Probability a cells will be conductive or not (Before cellular automata) | required")

    ccount_group = group.add_mutually_exclusive_group(required=True)
    ccount_group.add_argument('-j', '--conductive-material-range', dest='conductive_material_range', type=parse_tuple,
                    help=f"Range to randomly pick a number of conductive materials; max range = {len(conductive_indices)} | required")
    ccount_group.add_argument('-c', '--conductive-material-count', dest='conductive_material_count', type=int,
                    help=f"Static count of total conductive materials; max count = {len(conductive_indices)} | required")
    
    group.add_argument('-u', '--enable-absolute-permittivity', dest='enable_absolute_permittivity', action='store_true',
                    help="Enables converting material permittivity from relative to absolute | default: Off")
    group.add_argument('-l', '--enable-fixed-charges', dest='enable_fixed_charges', action='store_true',
                help="Enables solving for fixed charges, charges are free by default | default: Off")
    group.add_argument('-z', '--max-iterations', dest='max_iterations', type=int, default=3000,
                    help="Maximum allowed iterations to run electrostatic potential solvers | default: 3000")
    group.add_argument('-e', '--convergence-tolerance', dest='convergence_tolerance', type=float, default=1e-6,
                    help="Tolerance for convergence; reached when the maximum change between iterations falls below this value | default: 1E-6")
    group.add_argument('-w', '--save-states', dest='save_states', action='store_true',
                help="Enables saving states, where iteration is a power of two | default: Off")


def check_simulation_args(args):
    if not (4 < args.image_size < 1025):
        raise ValueError("IMAGE_SIZE must be a INT between [5, 1024]")
    
    if hasattr(args, 'conductive_material_ratio') and args.conductive_material_ratio is not None:
        (min_cond_ratio, max_cond_ratio) = util.compute_minimum_ratios(args.image_size, args.conductive_material_ratio)
        if not (min_cond_ratio <= args.conductive_material_ratio <= max_cond_ratio):
            ratio_range_str = f'equal to ({min_cond_ratio})' if min_cond_ratio == max_cond_ratio else f"between ({min_cond_ratio}, {max_cond_ratio})"
            raise ValueError(f"CONDUCTIVE_MATERIAL_RATIO must be a FLOAT {ratio_range_str}, enforcing at least 1 conductive material and 1 insolating material")

    if hasattr(args, 'conductive_material_prob') and args.conductive_material_ratio is not None:
        if not (0.0 <= args.conductive_material_prob <= 1.0):
            raise ValueError(f"CONDUCTIVE_MATERIAL_PROB must be a FLOAT, between [0.0,1.00]")
    
    num_conductive = len(conductive_indices)
    if hasattr(args, 'conductive_random_range') and args.conductive_random_range is not None:
        minr, maxr = args.conductive_random_range
        if not(0 < minr < maxr and minr < maxr < num_conductive):
            raise ValueError(f"CONDUCTIVE_RANDOM_RANGE must be a TUPLE(INT, INT) with value between [1, {num_conductive }]")

    if hasattr(args, 'conductive_static_count') and args.conductive_static_count is not None:
        if not 0 < args.conductive_static_count <= num_conductive:
            raise ValueError(f"CONDUCTIVE_STATIC_COUNT must be a INT between [1, {num_conductive}]")    

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


def add_output_group(parser):
    group = parser.add_argument_group('output path options')
    group.add_argument('-o', '--output-path', dest='output_path', type=str, default=util.path.dirname(util.path.abspath(__file__)),
                        help="Path the the directory to create [--output-folder] and save to | default: current directory")
    group.add_argument('-f', '--output-folder', dest='output_folder', type=str,
                        help="Output folder name to create and save simulation data to | default: esp_dataset")


def check_output_args(args, filename):
    if hasattr(args, 'output_path') and not util.path.exists(args.output_path):
        raise FileNotFoundError(f"OUTPUT_PATH '{args.output_path}' does not exist")
    if isinstance(args.output_folder, str) and not (0 < len(args.output_folder) < 129):
        raise ValueError(f"OUTPUT_FOLDER '{args.output_folder}' must have a length between [1, 128]")
    if filename == 'create_dataset.py' and args.output_folder is None:
        raise ValueError(f"OUTPUT_FOLDER '{args.output_folder}' is required for new datset creation")


def add_dataset_group(parser):
    group = parser.add_argument_group('dataset options')

    group.add_argument('-i', '--dataset-path', dest='dataset_path', type=str, default='.',
                        help="Path the the input dataset to read and process | default: current directory")

    group.add_argument('-w', '--disable-normalization', dest='normalize',  action='store_false', 
                    help="Option to disable normalizing simulation outputs with a min max scaler | default: Off")
                    
    group.add_argument('-v', '--simvp-format', dest='simvp_format', action='store_true',
                    help="Option to save dataset formatted for SimVP | default: Off")
    
    group.add_argument('-g', '--sample-plots', dest='sample_plots', type=int, default=0,
                    help="Optional number of samples to plot; No samples are plotted if set to '0' | default: 0")
    
    group.add_argument('-a', '--plot-states', dest='plot_states', action='store_true', 
                        help="Option to plot initial, intermediate, and final states; Requires passing [--save-states] to create_dataset.py; | default: Off")


def check_dataset_args(args):
    if hasattr(args, 'dataset_path') and not util.path.isfile(args.dataset_path) or '.hdf5' not in args.dataset_path:
        raise FileNotFoundError(f"DATASET_PATH '{args.dataset_path}' does not exist or is not a hdf5 dataset")
    
    seed_range = util.re.findall(r'\d+', util.path.basename(args.dataset_path).split('.')[0])
    max_plots = int(seed_range[-1])-int(seed_range[-2])+1

    if not (0 <= args.sample_plots <= max_plots):
            raise ValueError(f"SAMPLE_PLOTS '{args.sample_plots}' must be None or in range [1, {max_plots}]")
    
    if args.sample_plots == 0 and not args.simvp_format and not args.normalize:
        raise ValueError(f"No options are enabled to normalize, format, or plot samples...")


def check_args(parser, file_name):
    args = parser.parse_args()

    if "multiprocess" in executable_groups[file_name]:
        check_multiprocess_args(args)
    if "simulation" in executable_groups[file_name]:
        check_simulation_args(args)
    if "output" in executable_groups[file_name]:
        check_output_args(args, file_name)
    if "dataset" in executable_groups[file_name]:
        check_dataset_args(args)
    return args


def process_args(exe_file):
    file_name = util.path.basename(exe_file)
    parser = ap.ArgumentParser(description="Electro Static Potential Simulation")
    parser.add_argument('-d', '--debug', dest='debug_on', action='store_true', 
                        help="Enables logging with debug level verbosity | default: off")

    if "dataset" in executable_groups[file_name]:
        add_dataset_group(parser)
    if "multiprocess" in executable_groups[file_name]:
        add_multiprocess_group(parser, file_name)
    if "simulation" in executable_groups[file_name]:
        add_simulation_group(parser)
    if "output" in executable_groups[file_name]:
        add_output_group(parser)
    args = check_args(parser, file_name)

    return args