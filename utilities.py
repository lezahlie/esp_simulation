from setup_logger import setup_logger, logging, global_logger as logger
from json import loads, dump, load, dumps
from multiprocessing import Process, Manager, Lock, current_process
from os import makedirs, path, getpid, cpu_count, remove, listdir, environ, rename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as mcolor
import numpy as np
import pandas as pd
import re
import h5py
import warnings
import scipy.ndimage as ndimg
import numpy as np
import hashlib
from collections import deque, namedtuple
from glob import glob
from pprint import pprint

DEFAULT_DATAFILE_EXT = "hdf5"
SIMVP_DATAFILE_EXT = 'npy'
DATATYPE_NAME = "electrostatic"

def create_save_states_predicate(conditions):
    if 'all' in conditions:
        return lambda i: True

    combined_predicate = lambda i: False

    for condition in conditions:
        
        # Handle 'first-N'
        if isinstance(condition, tuple) and condition[0] == 'first':
            n = condition[1]
            # Combine with OR: (1 <= i <= n) or the existing combined_predicate
            combined_predicate = lambda i, combined_predicate=combined_predicate, n=n: (1 <= i <= n) or combined_predicate(i)

        # Handle 'interval-T'
        elif isinstance(condition, tuple) and condition[0] == 'interval':
            t = condition[1]
            # Combine with OR: (i % t == 0) or the existing combined_predicate
            combined_predicate = lambda i, combined_predicate=combined_predicate, t=t: (i % t == 0) or combined_predicate(i)

        # Handle 'base-N'
        elif isinstance(condition, tuple) and condition[0] == 'base':
            b = condition[1]
            def is_power(i, num):
                if i < 1:
                    return False
                while i % num == 0:
                    i //= num
                return i == 1
            # Combine with OR: is_power(i, b) or the existing combined_predicate
            combined_predicate = lambda i, combined_predicate=combined_predicate, b=b, is_power=is_power: is_power(i, b) or combined_predicate(i)

    return combined_predicate


def dynamic_precision_round(value):
    factor = 10 ** 2
    rounded_value = np.ceil(value * factor) / factor
    return rounded_value


def compute_nonboundnary_cells(image_size):
    return (image_size - 2) ** 2


def compute_minimum_ratios(image_size, conductive_material_ratio):
    total_internal_cells = (image_size - 2) ** 2
    material_cell_ratio=1.0

    material_cell_count = int(material_cell_ratio * total_internal_cells)
    
    # ensures at least 1 material is insolation
    max_conductive_ratio = dynamic_precision_round((material_cell_count - 1) / material_cell_count)
    effective_conductive_ratio = min(conductive_material_ratio, max_conductive_ratio)

    # ensures at least 1 material is conductive
    conductor_cell_count = max(1, int(effective_conductive_ratio * material_cell_count))
    min_conductive_ratio = max(effective_conductive_ratio, dynamic_precision_round(conductor_cell_count / material_cell_count))

    return (min_conductive_ratio, max_conductive_ratio)



# creates a folder path if it doesn't exist
def create_folder(folder_name):
    if path.isdir(folder_name):
        return path.abspath(folder_name)
    folder_path=path.dirname(path.abspath(__file__))
    dir_path = path.join(folder_path, folder_name)
    makedirs(dir_path, exist_ok=True)
    if not path.exists(dir_path):
        logger.error(f"Cannot create folder '{folder_name}' in path: {folder_path}", stacklevel=2)
    return dir_path


# returns a path to a file, creates the the root directory if needed
def create_file_path(folder_path, file_name):
    out_path = create_folder(folder_path)
    file_path = path.join(out_path, file_name)
    return file_path


# removes a file if it exists
def remove_if_exists(file_path=None):
    if file_path and path.exists(file_path):
        remove(file_path) 


# converts dataframes to lists of dictionary records and vice versa
def dataframe_dictrecord_converter(data):
    if isinstance(data, pd.DataFrame):
        dict_record = data.to_dict(orient='records')
        return dict_record
    elif isinstance(data, list) and isinstance(data[0], dict):
        df = pd.DataFrame(data)
        return df
    else:
        raise TypeError(f"Expected types 'pd.Dataframe' or 'dict[]' for conversion, but recieved type '{type(data)}'")


# splits ranges of seeds based on step size, meant for data splitting between tasks
def split_seed_range(seed_range, seed_step):
    min_seed, max_seed = seed_range
    return [(start_seed, min(start_seed + seed_step - 1, max_seed)) for start_seed in range(min_seed, max_seed + 1, seed_step)]


# flattens nested dictionaries, by appending top level key as a prefix
def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# shared dictionary between processes
def update_shared_data(new_data, shared_data, shared_lock):
    with shared_lock:
        shared_data_copy = shared_data.copy()

    for category in ['image', 'metric']:
        if category not in new_data:
            continue
        if category not in shared_data_copy:
            shared_data_copy[category] = {}

        for key, new_stats in new_data[category].items():
            if key not in shared_data_copy[category]:
                shared_data_copy[category][key] = new_stats
            else:
                original_stats = shared_data_copy[category][key]

                # Calculate new min and max
                new_min = min(original_stats['min'], new_stats["min"])
                new_max = max(original_stats['max'], new_stats["max"])

                # Calculate the new count (total number of elements)
                new_count = original_stats['count'] + new_stats["count"]

                # Calculate the pooled mean
                new_mean = (original_stats["count"] * original_stats["mean"] + new_stats["count"] * new_stats["mean"]) / new_count

                # Calculate the pooled variance
                pooled_variance = (
                    (original_stats["count"] - 1) * original_stats["std"] ** 2 +
                    (new_stats["count"] - 1) * new_stats["std"] ** 2 +
                    (original_stats["count"] * new_stats["count"]) / (new_count) * (original_stats["mean"] - new_stats["mean"]) ** 2
                ) / (new_count - 1)

                # Calculate the pooled standard deviation
                new_std = np.sqrt(pooled_variance)

                new_shape = [original_stats['shape'][0] + new_stats['shape'][0]] + original_stats['shape'][1:]

                # Update the shared data dictionary with pooled values
                shared_data_copy[category][key] = {
                    'min': new_min,
                    'max': new_max,
                    'mean': new_mean,
                    'std': new_std,
                    'count': new_count,
                    'shape': new_shape
                }

    with shared_lock:
        shared_data.update(shared_data_copy)

def compute_local_stats(sim_results_list):
    stats_dict = {}
    data_groups = ['image', 'metric']  # Track image and metric data

    for sim_results in sim_results_list:
        for group in data_groups:
            if group not in sim_results:
                continue

            if group not in stats_dict:
                stats_dict[group] = {}

            for key, value in sim_results[group].items():
                data = np.array(value)
                local_min = np.min(data)
                local_max = np.max(data)
                local_mean = np.mean(data)
                local_std = np.std(data)
                local_size = data.size 
                local_shape = list(data.shape) if group == "image" else [1]

                if key not in stats_dict[group]:
                    stats_dict[group][key] = {
                        'min': local_min,
                        'max': local_max,
                        'mean': local_mean,
                        'std': local_std,
                        'count': local_size,
                        'shape': [1] + local_shape
                    }
                else:
                    stats_dict[group][key]['min'] = min(stats_dict[group][key]['min'], local_min)
                    stats_dict[group][key]['max'] = max(stats_dict[group][key]['max'], local_max)
                    stats_dict[group][key]['mean'] = local_mean
                    stats_dict[group][key]['std'] = local_std
                    stats_dict[group][key]['count'] += local_size
                    stats_dict[group][key]['shape'][0] += 1

    return stats_dict

def extract_minmax_tuples(global_stats):
    min_max_dict = {}
    for category, values in global_stats.items():
        min_max_dict[category] = {}
        for key, stats in values.items():
            min_max_dict[category][key] = (stats["min"], stats["max"])
    return min_max_dict

# helper to min max normalize 1 or 2 dim data
def normalize(data, data_minmax:tuple[float,float], scaler_range:tuple[float,float]=(0.0, 1.0), inverse=False):
    min_out, max_out = scaler_range
    min_val, max_val= data_minmax
    if min_val >= max_val:
        raise TypeError(f"min_val {min_val} must be less than max val {max_val}")
    normalized = (data - min_val) / (max_val - min_val)
    if inverse:
        unscaled_channel = normalized * (max_val - min_val) + min_val
        return unscaled_channel
    else:
        scaled_channel = min_out + normalized * (max_out - min_out)
        return scaled_channel


# scales numpy arrays to a specified range, works for 1, 2, and 3 dims
def minmax_scaler(data, minmax_values: tuple[int, int], scaler_range=(0.0, 1.0), inverse=False):
    if scaler_range[1] > 0 and scaler_range[0] >= scaler_range[1]:
        raise ValueError(f"min_val {scaler_range[0]} must be less than max_val {scaler_range[1]}")
    
    if np.isscalar(data):
        return data if data == 0 else normalize(data, minmax_values, scaler_range, inverse)
    elif not isinstance(data, np.ndarray):
        try:
            data = np.array(data)
        except Exception as e:
            raise TypeError(f"Failed converting input type {type(data)} to a NumPy array")
        
    if 0 < data.ndim < 3:  # 1D (L), 2D (W x H)
        normalized_data = normalize(data, minmax_values, scaler_range, inverse)
    elif data.ndim == 3:   # 3D (C x W x H)
        normalized_data = np.zeros_like(data, dtype=np.float32)
        for channel in range(data.shape[0]):
            normalized_data[channel]= normalize(data[channel], minmax_values, scaler_range, inverse)
    elif data.ndim == 4:   # 4D (B x C x W x H)
        for batch in range(data.shape[0]):
            for channel in range(data.shape[1]):
                normalized_data[batch, channel] = normalize(data[batch, channel], minmax_values, scaler_range, inverse)
    else:
        raise ValueError(f"Input shape '{data.shape}' must be one of: 1D (L), 2D (W x H), 3D (C x W x H), 4D (B x C x W x H)")

    return normalized_data


# serializes numpy types bc ya know
def serialize_numpy_types(obj):
    if isinstance(obj, dict):
        return {key: serialize_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_numpy_types(element) for element in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    else:
        return obj


# writes or appends to a csv file
def save_to_json(file_path, content_in:dict, mode='w', indent=4):
    try:
        content_out = serialize_numpy_types(content_in)
        with open(file_path, mode) as json_file:
            dump(content_out, json_file, indent=indent)
    except Exception as e:
        logger.error(e, stacklevel=2)   


# reads json file
def read_from_json(file_path):
    try:
        with open(file_path, 'r') as json_file:
            content = load(json_file)
        return content
    except Exception as e:
        logger.error(e, stacklevel=2)   


# REF: https://github.com/drewg02/OpenSTL/blob/master/SimVP_Standalone/simvp_standalone/experiment_recorder.py
def generate_unique_id(expe_record):
    """Generate a SHA-256 hash as a unique ID for the experiment record."""
    if isinstance(expe_record, np.ndarray):
        record = expe_record.tolist()
    else:
        record = expe_record

    if not isinstance(record, list):
        raise TypeError(f"Expected experiiment record to be a numpy array or a list")
    
    serialized_record = dumps(record, sort_keys=True)
    hash_object = hashlib.sha256(serialized_record.encode())
    return hash_object.hexdigest()


# saves numpy files formated for SimVP
def save_to_numpy(images: tuple[np.ndarray, np.ndarray], data_path:str, datatype:str, sim_idx:int):
    unique_id = generate_unique_id(images[0].tolist())
    folder_name = f"{unique_id}_{datatype}_{sim_idx}"
    save_path = create_folder(path.join(data_path, folder_name))
    for i, image in enumerate(images):
        image_path = f"{save_path}/{i}.npy"
        np.save(image_path, image)
        logger.info(f"Saved image {i} to '{image_path}'", stacklevel=2)
        
    # for i, _ in enumerate(images):
    #     image = np.load(f"{save_path}/{i}.npy")
    #     assert np.array_equal(images[i], image), f"images {i} do not match!"

    return save_path


# writes or appends to a hdf5 file
def save_to_hdf5(data_dict_list, file_path, chunk_size=None, flatten=False):
    def write_data_to_group(group, data):
        for key, value in data.items():
            if isinstance(value, (int, float)):
                group.attrs[key] = value
                #logger.debug(f"Saved attribute: {key} => {value}", stacklevel=2)
            elif isinstance(value, str):
                string_dt = h5py.string_dtype(encoding='utf-8', stacklevel=2)
                group.create_dataset(key, data=value, dtype=string_dt)
                #logger.debug(f"Saved string dataset: {key} => {value}", stacklevel=2)
            elif isinstance(value, np.ndarray):
                group.create_dataset(key, data=value)
                #logger.debug(f"Saved array dataset: {key} => array with shape {value.shape}", stacklevel=2)
            elif isinstance(value, dict):
                subgroup = group.create_group(key)
                write_data_to_group(subgroup, value) 
            else:
                raise TypeError(f"Invalid type for {key}: {type(value)}. Expected int, float, str, or np.ndarray.")
            
    chunk = chunk_size or 1
    mode = 'a' if path.exists(file_path) else 'w'  
    try:
        with h5py.File(file_path, mode) as f:
            existing_indices = [int(k.split('_')[1]) for k in f.keys() if k.startswith("record_")]
            current_max_index = max(existing_indices) + 1 if existing_indices else 0
            total_records = len(data_dict_list)
            logger.debug(f"Saving {total_records} records to file starting at index {current_max_index}", stacklevel=2)

            for i in range(0, total_records, chunk):
                for idx in range(i, min(i + chunk, total_records)):
                    record_index = current_max_index + idx
                    record_group_name = f"record_{record_index}"

                    if record_group_name in f:
                        logger.debug(f"Skipping existing group: {record_group_name}", stacklevel=2)
                        continue

                    record_dict = flatten_dict(data_dict_list[idx]) if flatten else data_dict_list[idx]
                    record_group = f.create_group(record_group_name)
                    write_data_to_group(record_group, record_dict)
                    logger.debug(f"Created group: {record_group_name}", stacklevel=2)

    except (Exception, OSError, IOError, TypeError) as e:
        logger.error(f"Error writing to HDF5 file {file_path}: {e}", stacklevel=2)


# read a hdf5 file in or a random # of samples 
def read_from_hdf5(file_path, sample_size=None, chunk_size=None, flatten=True):
    def load_group_data(group):
        group_dict = {}
        group_dict.update({k: v for k, v in group.attrs.items()})
        #logger.debug(f"Loaded attributes: {group.attrs.items()}", stacklevel=2)
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                subgroup_data = load_group_data(item)
                if flatten:
                    group_dict.update(flatten_dict(subgroup_data, parent_key=key))
                else:
                    group_dict[key] = subgroup_data
            else:
                group_dict[key] = item[:]
                #logger.debug(f"Loaded dataset: {key} => shape {item[:].shape}", stacklevel=2)
        return group_dict
    
    chunk = chunk_size or 1
    data_dict_list = []
    try:
        with h5py.File(file_path, 'r') as f:
            all_keys = list(f.keys())
            if sample_size and 0 < sample_size < len(all_keys):
                selected_keys = np.random.choice(all_keys, sample_size, replace=False)
            else:
                selected_keys = all_keys

            for i in range(0, len(selected_keys), chunk):
                chunk_keys = selected_keys[i:i + chunk]
                for key in chunk_keys:
                    group = f[key]
                    data = load_group_data(group)
                    data_dict_list.append(data)
            return data_dict_list

    except (OSError, IOError, TypeError) as e:
        logger.error(f"Cannot read from HDF5 file: {file_path} due to: {e}", stacklevel=2)


# copies attributes from a HDF5 file
def _copy_attributes(src, dst):
    for attr_key, attr_value in src.attrs.items():
        dst.attrs[attr_key] = attr_value
        logger.debug(f"Copied attribute for {src.name}: {attr_key} => {attr_value}")


# copies datasets from a HDF5 file
def _copy_dataset(src_dataset, dst_group, dataset_name):
    data = src_dataset[:]
    dst_dataset = dst_group.create_dataset(dataset_name, data=data)
    _copy_attributes(src_dataset, dst_dataset)


# copies groups from a HDF5 file
def _copy_group(src_group, dst_group, group_name_prefix=""):
    for key in src_group:
        new_group_name = f"{group_name_prefix}_{key}" if group_name_prefix else key
        item = src_group[key]

        if isinstance(item, h5py.Group):
            dst_subgroup = dst_group.create_group(new_group_name)
            _copy_attributes(item, dst_subgroup)
            _copy_group(item, dst_subgroup)
        elif isinstance(item, h5py.Dataset):
            _copy_dataset(item, dst_group, new_group_name)
        else:
            logger.warning(f"Unexpected item type: {type(item)} for key '{key}'", stacklevel=2)


# reads in a chunk of records as time to save memory
def _get_record_chunk(src_file, chunk_size: int):
    chunk = []
    for record_name, record in src_file.items():
        if isinstance(record, h5py.Group):
            chunk.append((record_name, record))
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        else:
            logger.warning(f"'{record_name}' is not a group. Skipping.", stacklevel=2)
    if chunk:
        yield chunk


# combines HDF5 files into a single file, meant for parallel IO
def combine_hdf5_files(input_file_paths, output_file_path, chunk_size=None):
    chunk = chunk_size or 1
    try:   
        with h5py.File(output_file_path, 'a') as dst_file:
            record_index = len(dst_file.keys())
            
            for file_path in input_file_paths:
                with h5py.File(file_path, 'r') as src_file:
                    for record_name in src_file.keys():
                        logger.debug(f"Combining record: {record_name}")

                        unique_record_name = f"record_{record_index}"
                        dst_group = dst_file.create_group(unique_record_name)

                        _copy_attributes(src_file[record_name], dst_group)
                        _copy_group(src_file[record_name], dst_group)

                        record_index += 1
                remove_if_exists(file_path)
                
    except (OSError, IOError, TypeError) as e:
        logger.error(f"Cannot read HDF5 files and combine to one HDF5 file due to: {e}", stacklevel=2)

    
# normalizes existing HDF5 data to a new hdf5 data file
def normalize_hdf5_to_hdf5(input_file_path: str, output_file_path: str, stats_values_dict: dict | None,   
                            chunk_size: int | None = None, scaler_range=(0.0, 1.0), inverse=False):
    try:
        minmax_values = extract_minmax_tuples(stats_values_dict)
        with h5py.File(input_file_path, 'r') as src_file, h5py.File(output_file_path, 'w') as dst_file:
            for chunk in _get_record_chunk(src_file, chunk_size or 1):
                for record_name, record in chunk:
                    dst_group = dst_file.create_group(record_name)
                    _copy_attributes(record, dst_group)
                    
                    for group_key, group in record.items():
                        if isinstance(group, h5py.Group):
                            dst_subgroup = dst_group.create_group(group_key)
                            _copy_attributes(group, dst_subgroup)

                            for dataset_key, dataset in group.items():
                                data = dataset[:]
                                if group_key in minmax_values and dataset_key in minmax_values[group_key]:
                                    minmax_val = minmax_values[group_key][dataset_key]
                                    data = minmax_scaler(data, minmax_val, scaler_range, inverse)
                                dst_subgroup.create_dataset(dataset_key, data=data)
                            
                            for attr_key, attr_value in group.attrs.items():
                                value = attr_value
                                if group_key in minmax_values and attr_key in minmax_values[group_key]:
                                    minmax_val = minmax_values[group_key][attr_key]
                                    value = minmax_scaler(value, minmax_val, scaler_range, inverse)
                                dst_subgroup.attrs[attr_key] = value
                                logger.debug(f"Normalized attribute for {group.name}: {attr_key} => {value}")

    except (OSError, IOError, TypeError) as e:
        logger.error(f"Cannot read and normalize data from HDF5 file due to: {e}", stacklevel=2)


def _process_image_stack(group, keys, minmax_values, normalize, scaler_range, inverse, record_name):
    images = []

    for key in keys:
        if key not in group:
            logger.warning(f"Key '{key}' not found in group of record '{record_name}', skipping.", stacklevel=2)
            continue

        data = group[key][:]
        if normalize and key in minmax_values:
            data = minmax_scaler(data, minmax_values[key], scaler_range, inverse)

        images.append(data)

    if len(images) != len(keys):
        logger.warning(f"Missing some images in record '{record_name}', skipping.", stacklevel=2)
        return None
    
    images_stacked = np.stack(images, axis=0).astype(float)
    return images_stacked


# normalizes hdf5 data and writes to numpy files formatted for SimVP
def normalize_hdf5_to_numpy(input_file_path: str, stats_values_dict: dict | None,  
                            data_path: str, datatype: str, normalize: bool = True, chunk_size: int|None = None,  
                            inverse: bool = False, scaler_range: tuple[float, float] = (0.0, 1.0)):

    minmax_values_dict = extract_minmax_tuples(stats_values_dict)
    subgroup = 'image'
    minmax_values = minmax_values_dict[subgroup]
    input_keys = ["potential_state_initial", "charge_distribution", "permittivity_map"]
    output_keys = ["potential_state_final"]
    sim_idx = 0

    try:
        with h5py.File(input_file_path, 'r') as src_file:

            for chunk in _get_record_chunk(src_file, chunk_size or 1):
                for record_name, record in chunk:
                    if "image" not in record:
                        logger.warning(f"Group 'image' not found in record '{record_name}', skipping.", stacklevel=2)
                        continue

                    image_group = record[subgroup]

                    input_images_stacked = _process_image_stack(image_group, input_keys, minmax_values, 
                                                                normalize, scaler_range, inverse, record_name)

                    output_images_stacked = _process_image_stack(image_group, output_keys, minmax_values, 
                                                                normalize, scaler_range, inverse, record_name)

                    if input_images_stacked is None or output_images_stacked is None:
                        continue

                    logger.debug(f"input_images.shape: {input_images_stacked.shape}, output_images.shape: {output_images_stacked.shape}")

                    save_to_numpy((input_images_stacked, output_images_stacked), data_path, datatype, sim_idx)
                    sim_idx += 1


        channel_stats = {
            'input_channels': {
                i: {'name': key, **{k: v for k, v in stats_values_dict['image'][key].items() if k not in ['count', 'shape']}}
                for i, key in enumerate(input_keys)
            },
            'output_channels': {
                i: {'name': key, **{k: v for k, v in stats_values_dict['image'][key].items() if k not in ['count', 'shape']}}
                for i, key in enumerate(output_keys)
            }
        }

        return channel_stats

    except (OSError, IOError, TypeError) as e:
        logger.error(f"Cannot normalize data from HDF5 file and save to NumPy files due to: {e}", stacklevel=2)
