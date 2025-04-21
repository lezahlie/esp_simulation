from setup_logger import setup_logger
logger = setup_logger(__file__, log_stdout=True, log_stderr=True)
from utilities import path, re, np, mtick, mcolor, plt, namedtuple, create_file_path
from electrostatic_mappers import PERMITTIVITY_TYPE, material_category_names

def determine_precision(cbar_ticks, default_precision=3):
    if len(cbar_ticks) < 2:
        return default_precision
    
    tick_diffs = np.diff(sorted(cbar_ticks))
    min_diff = np.min(tick_diffs[tick_diffs > 0]) if np.any(tick_diffs > 0) else 1
    precision = max(default_precision, -int(np.floor(np.log10(min_diff))))

    while len(set(f"{tick:.{precision}f}" for tick in cbar_ticks)) < len(cbar_ticks):
        precision += 1
    return precision

def dynamic_tick_formatter(val, pos=None, precision=3):
    if val == 0:
        return "0"
    if val <= 1e-4 or val >= 1e4:
        return f'{val:.{precision}e}'
    return f'{val:.{precision}f}'


def plot_image_map(ax, array, cbar_name, cmap_name='plasma', discrete=False, extrema_values=None, tick_labels=None, threshold=1e3):
    if isinstance(extrema_values, list):
        min_value, max_value = tuple(extrema_values)
    elif isinstance(extrema_values, tuple):
        min_value, max_value = extrema_values 
    else:
        min_value, max_value = np.min(array), np.max(array)

    label_rotate=0

    if discrete:
        if tick_labels is not None:
            tick_count = len(tick_labels)
            cbar_ticks = np.arange(tick_count)
            cbar_formatter = mtick.FuncFormatter(lambda val, pos: tick_labels[int(val)] if int(val) < len(tick_labels) else "")
            label_rotate=-45
        else:
            tick_count = int(max_value) + 1 
            cbar_ticks = np.arange(tick_count)
            show_ticks = cbar_ticks[::2] if tick_count > 12 else cbar_ticks
            norm = mcolor.BoundaryNorm(cbar_ticks, tick_count)
            cbar_formatter = mtick.FuncFormatter(lambda val, pos: int(val) if val in show_ticks else "")
            cbar_ticks = show_ticks
        custom_cmap = plt.get_cmap(cmap_name, tick_count)
        norm = None

    else:
        diff_value = max_value / min_value if min_value > 0 else max_value
        if diff_value > threshold and min_value > 0:
            norm = mcolor.LogNorm(vmin=min_value, vmax=max_value)
            cbar_ticks = np.logspace(np.log10(min_value), np.log10(max_value), num=10)
        else:
            norm = plt.Normalize(vmin=min_value, vmax=max_value)
            cbar_ticks = np.linspace(min_value, max_value, num=10)

        custom_cmap = plt.get_cmap(cmap_name)
        cbar_formatter = mtick.FuncFormatter(lambda val, pos: dynamic_tick_formatter(val, pos))

    im = ax.imshow(array, cmap=custom_cmap, norm=norm, origin='upper')

    cbar = plt.colorbar(im, shrink=1.0, ax=ax)
    cbar.set_ticks(cbar_ticks)
    cbar.ax.yaxis.set_major_formatter(cbar_formatter)
    cbar.ax.yaxis.set_tick_params(rotation=label_rotate)
    cbar.ax.set_ylabel(cbar_name)

    return ax


def plot_sample_images(map_list, plot_title, plot_path, nrows = 1):
    if len(map_list) > 4:
        nrows = 2
    ncols = len(map_list)//nrows
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*4.5, nrows*4))
    axs = axs.flatten()

    for ax, config in zip(axs, map_list):
        ax = plot_image_map(
            ax=ax,
            array=config.array,
            cbar_name=config.cbar_label,
            cmap_name=config.cmap_name,
            extrema_values=config.extrema_values,
            discrete=config.discrete,
            tick_labels=config.tick_labels
        )
        ax.set_title(config.title.replace('_', ' '))
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axs.flat:
        if not ax.has_data():
            ax.axis('off')
            
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(plot_title, y=0.975)
    plt.savefig(plot_path, format="png")
    plt.close()


def plot_simulation_samples(sample_dicts:list[dict], plot_path:str, plot_prefix:str, global_extrema_values:dict|None=None, plot_states:bool=False):
    for sample in sample_dicts:
        permittivity_titles = f"{PERMITTIVITY_TYPE.title()} Permittivity", "Permittivity"
        if PERMITTIVITY_TYPE == 'absolute':
            permittivity_titles[0]+=" (Farads/meters)"
            permittivity_titles[1]+=" (F/m)"

        MapConfig = namedtuple("MapConfig", ["array", "title", "cbar_label", "cmap_name", "extrema_values", "discrete", "tick_labels"])
        extrema_values = global_extrema_values if global_extrema_values is not None else {}
        input_minmax = extrema_values.get('initial_potential_map', None)
        output_minmax = extrema_values.get('final_potential_map', None)
        perm_minmax = extrema_values.get('permittivity_map', None)
        charge_minmax = extrema_values.get('charge_distribution', None)

        map_list = [
                MapConfig(sample['image_permittivity_map'], f"{permittivity_titles[0]}", 
                        permittivity_titles[1], 'plasma', perm_minmax, False, None),
                MapConfig(sample['image_charge_distribution'], "Charge Distribution (Coulombs/meters)", 
                        "Charge Density (C/m)", 'RdYlBu_r', charge_minmax, False, None),
                MapConfig(sample['image_initial_potential_map'], "Initial Potential (Volts)", 
                        "Potential (V)", 'turbo', input_minmax, False, None),
                MapConfig(sample['image_final_potential_map'], "Final Potential (Volts)", 
                        "Potential (V)", 'turbo', output_minmax, False, None)
            ]

        final_state = 'stopped' if sample['meta_converged']==0 else 'converged'
        random_seed = sample['meta_random_seed']
        max_delta = sample['meta_max_delta']
        total_iterations = sample['meta_total_iterations']
        plot_title = f"Seed[{random_seed}]: solver {final_state} with max_delta = {max_delta:g} after {total_iterations} iterations" 

        plot_folder = path.join(plot_path, f"sample_{random_seed}")
        plot_file =  create_file_path(plot_folder, f"{plot_prefix}_{random_seed}.png")
        plot_sample_images(map_list, plot_title, plot_file)
        logger.info(f"Saved sample images plot for seed {random_seed} to: {plot_file}")
        
        if plot_states:
            new_plot_file =  plot_file.replace(".png", "_states.png")
            potential_states = {0: sample['image_initial_potential_map']}
            for key, val in sample.items():
                if key.startswith("image_potential_state_"):
                    iteration = int(key.split("_")[-1])
                    potential_states[iteration] = val

            potential_states[total_iterations] = sample['image_final_potential_map']
            plot_potential_states(potential_states, new_plot_file, random_seed)
            logger.info(f"Saved sample states plot for seed {random_seed} to: {new_plot_file}")


def plot_potential_states(potential_states, plot_path, seed_num, normalized=False):
    states = potential_states
    global_min = np.min([np.min(state) for state in states.values()])
    global_max = np.max([np.max(state) for state in states.values()])
    data_min, data_max = round(global_min), round(global_max)

    if normalized:  
        if global_max - global_min != 0:
            states = {k: (state - global_min) / (global_max - global_min)
                        for k, state in potential_states.items()}
        data_min, data_max = 0, 1

    ## this is for when all states are saved, but that's not implemented right now
    ## currently saving states: initial, {1, ..., 20}, {log2(i) < final}, final 

    # filtered_states = {
    #     k: v
    #     for k, v in states.items()
    #     if k == 0 or (k < 1024 and (k & (k - 1)) == 0) or k == max_key
    # }
    # sorted_states = dict(sorted(filtered_states.items()))

    sorted_states = dict(sorted(states.items()))
    num_states = len(sorted_states)


    if num_states > 6:
        ncols = 7 
        nrows = (num_states + ncols - 1) // ncols
        
    else:
        ncols = num_states
        nrows = 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5))
    axes = axes.flatten() 

    for i, (step, state) in enumerate(sorted_states.items()):
        ax = axes[i]
        im = ax.imshow(state, cmap='turbo', origin='upper', vmin=data_min, vmax=data_max)
        ax.set_title(f"State #{step}", fontsize=20, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Potental (V)", labelpad=10, fontsize=20)
        tick_values = np.linspace(data_min, data_max, num=10)
        cbar.set_ticks(tick_values)
        cbar.ax.set_yticklabels([f"{tick:.1f}" for tick in tick_values])
        cbar.ax.yaxis.set_tick_params(labelsize=18, pad=5)   

    for j in range(num_states, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"Electrostatic Potential Simulation Time-Steps (Seed: {seed_num})", fontsize=22, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(plot_path)
    plt.close()


def plot_individual_images(map_list, plot_folder, seed_num):
    for config in map_list:
        fig, ax = plt.subplots(figsize=(5.5, 4.5)) 
        ax = plot_image_map(
            ax=ax,
            array=config.array,
            cbar_name=config.cbar_label,
            cmap_name=config.cmap_name,
            extrema_values=config.extrema_values,
            discrete=config.discrete,
            tick_labels=config.tick_labels
        )

        
        title_loc = 'left' if "C:" in config.title else 'center'
        ax.set_title(config.title, fontsize=16, loc=title_loc, pad=20)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"(Seed: {seed_num})", fontsize=14, labelpad=10)

        if not ax.has_data():
            ax.axis('off')

        sanitized_title = config.title
        if ':' in sanitized_title:
            sanitized_title = config.title.split(':', 1)[1].strip()

        sanitized_title = re.sub(r"\(.*?\)", "", sanitized_title).strip()
        sanitized_title = sanitized_title.replace(" ", "_").lower()

        plot_path = f"{plot_folder}/{sanitized_title}.png"
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.savefig(plot_path)
        plt.close()
