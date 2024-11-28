from setup_logger import setup_logger
logger = setup_logger(__file__, log_stdout=True, log_stderr=True)
from utilities import np, mtick, mcolor, plt
from electrostatic_mappers import PERMITTIVITY_TYPE, material_category_names

def determine_precision(cbar_ticks, default_precision=3):
    tick_diffs = np.diff(cbar_ticks)
    if len(tick_diffs) > 0:
        min_diff = np.min(tick_diffs[tick_diffs > 0]) if np.any(tick_diffs > 0) else 1
        relative_diff = min_diff / np.max(cbar_ticks)  
        precision = max(default_precision, -int(np.floor(np.log10(relative_diff))))  # Base precision
    else:
        precision = default_precision 
    
    while len(set(f"{tick:.{precision}g}" for tick in cbar_ticks)) < len(cbar_ticks):
        precision += 1  
    
    return precision


def dynamic_tick_formatter(val, pos=None, precision=3):
    if val == 0:
        return "0"
    if val <= 1e-4 or val >= 1e4:
        return f'{val:.{precision}e}'
    
    return f'{val:.{precision}f}'


def plot_image_map(ax, matrix, cbar_name, cmap='plasma', discrete=False, extrema_values=None, tick_labels=None, threshold=1e3):
    min_value, max_value = extrema_values if isinstance(extrema_values, tuple) else (np.min(matrix), np.max(matrix))
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
        custom_cmap = plt.get_cmap(cmap, tick_count)
        norm = None

    else:
        diff_value = max_value / min_value if min_value > 0 else max_value
        if diff_value > threshold and min_value > 0:
            norm = mcolor.LogNorm(vmin=min_value, vmax=max_value)
            cbar_ticks = np.logspace(np.log10(min_value), np.log10(max_value), num=10)
        else:
            norm = plt.Normalize(vmin=min_value, vmax=max_value)
            cbar_ticks = np.linspace(min_value, max_value, num=10)

        custom_cmap = plt.get_cmap(cmap)
        cbar_formatter = mtick.FuncFormatter(lambda val, pos: dynamic_tick_formatter(val, pos))

    im = ax.imshow(matrix, cmap=custom_cmap, norm=norm, origin='upper')

    cbar = plt.colorbar(im, shrink=1.0, ax=ax)
    cbar.set_ticks(cbar_ticks)
    cbar.ax.yaxis.set_major_formatter(cbar_formatter)
    cbar.ax.yaxis.set_tick_params(rotation=label_rotate)
    cbar.ax.set_ylabel(cbar_name)

    return ax


def plot_sample_images(map_list, plot_title, plot_path, nrows=2):
    ncols = len(map_list)//nrows
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*5, (nrows*4)+0.5))
    axs = axs.flatten()

    for ax, (matrix, title, cbar_name, cmap, extrema_values, discrete, tick_labels) in zip(axs, map_list):
        ax = plot_image_map(ax, matrix, cbar_name=cbar_name, cmap=cmap, extrema_values=extrema_values, discrete=discrete, tick_labels=tick_labels)
        ax.set_title(title.replace('_', ' '))
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axs.flat:
        if not ax.has_data():
            ax.axis('off')
            
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(plot_title, y=0.975)
    plt.savefig(plot_path)
    plt.close()


def plot_simulation_samples(sample_dicts:list[dict], global_minmax_values:dict, plot_path_prefix:str, solver_images_only:bool):
    for sample in sample_dicts:
        permittivity_titles = f"{PERMITTIVITY_TYPE.title()} Permittivity", "Permittivity"
        if PERMITTIVITY_TYPE == 'absolute':
            permittivity_titles[0]+=" (Farads/meters)"
            permittivity_titles[1]+=" (F/m)"
        
        image_extrema = global_minmax_values['image']
        if solver_images_only:
            map_list = [
                (sample['image_permittivity_map'], permittivity_titles[0], permittivity_titles[1], 'plasma', tuple(image_extrema['permittivity_map']), False, None),
                (sample['image_charge_distribution'], "Charge Distribution (Coulombs/meters)", "Charge Density (C/m)", 'RdYlBu_r',  tuple(image_extrema['charge_distribution']), False, None),
                (sample['image_initial_potential_map'], "Initial Potential Map (Volts)", "Potential (V)", 'turbo',  tuple(image_extrema['initial_potential_map']), False, None),
                (sample['image_final_potential_map'], "Final Potential Map (Volts)", "Potential (V)", 'turbo',  tuple(image_extrema['final_potential_map']), False, None)
            ]
        else:
            map_list = [
                (sample['mask_material_category_map'], "Material Category Mask", "Category", 'brg', None, True, material_category_names),
                (sample['mask_material_id_map'], "Material Id Map", "Id #", 'tab20b', None, True, None),  
                (sample['image_initial_potential_map'], "Initial Potential Map (Volts)", "Potential (V)", 'turbo',  tuple(image_extrema['initial_potential_map']), False, None),
                (sample['image_charge_distribution'], "Charge Distribution (Coulombs/meters)", "Charge Density (C/m)", 'RdYlBu_r',  tuple(image_extrema['charge_distribution']), False, None),
                (sample['image_permittivity_map'], permittivity_titles[0], permittivity_titles[1], 'plasma', tuple(image_extrema['permittivity_map']), False, None),
                (sample['image_final_potential_map'], "Final Potential Map (Volts)", "Potential (V)", 'turbo',  tuple(image_extrema['final_potential_map']), False, None)
            ]
            # map_list = [
            #     (sample['mask_material_category_map'], "Material Category Mask", "Category", 'brg', None, True, material_category_names),
            #     (sample['image_permittivity_map'], permittivity_titles[0], permittivity_titles[1], 'plasma', tuple(image_extrema['permittivity_map']), False, None),
            #     (sample['image_charge_distribution'], "Charge Distribution (Coulombs/meters)", "Charge Density (C/m)", 'RdYlBu_r', tuple(image_extrema['charge_distribution']), False, None),
            #     (sample['image_initial_potential_map'], "Initial Potential Map (Volts)", "Potential (V)", 'turbo',  tuple(image_extrema['initial_potential_map']), False, None),
            #     (sample['mask_material_id_map'], "Material Id Map", "Id #", 'tab20b', None, True, None),  
            #     (sample['image_final_potential_map'], "Final Potential Map (Volts)", "Potential (V)", 'turbo', tuple(image_extrema['final_potential_map']), False, None),
            #     (sample['image_electric_field_magnitude'], "Electric Field Magnitude (Volts/meters)", "Field (V/m)", 'inferno', tuple(image_extrema['electric_field_magnitude']), False, None)
            # ]

        final_state = 'stopped' if sample['meta_converged']==0 else 'converged'
        random_seed = sample['meta_random_seed']
        max_delta = sample['meta_max_delta']
        total_iterations = sample['meta_total_iterations']
        plot_title = f"Seed[{random_seed}]: solver {final_state} with max_delta = {max_delta:g} after {total_iterations} iterations" 

        plot_path  = f"{plot_path_prefix}_{random_seed}.png"
        plot_sample_images(map_list, plot_title, plot_path)
        logger.info(f"Saved seed {random_seed} plot to: {plot_path}")
