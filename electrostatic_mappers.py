from setup_logger import setup_logger
logger = setup_logger(__file__, log_stderr=True, log_stdout=True)
from utilities import np

epsilon_0 = 8.854187817e-12

material_properties = {
    'free': {
        'air_vacuum': {'index': 0, 'relative': 1.0}
    },
    'insulated': {
        'teflon': {'index': 1, 'relative': 2.1},
        'polyethylene': {'index': 2, 'relative': 2.4},
        'nylon': {'index': 3, 'relative': 3.6},
        'epoxy_resin': {'index': 4, 'relative': 4.2},
        'FR4': {'index': 5, 'relative': 4.8},
        'glass': {'index': 6, 'relative': 6.5},
        'rubber': {'index': 7, 'relative': 9.5}
    },
    'conductive': {
        'aluminum': {'index': 8, 'relative': 3.5},
        'nickel': {'index': 9, 'relative': 5.0},
        'stainless_steel': {'index': 10, 'relative': 6.2},
        'bronze': {'index': 11, 'relative': 8.0},
        'copper_alloy': {'index': 12, 'relative': 10.0},
        'zinc': {'index': 13, 'relative': 12.5},
        'tin': {'index': 14, 'relative': 15.0},
        'lead': {'index': 15, 'relative': 18.0},
        'graphite': {'index': 16, 'relative': 22.0},
        'silicon': {'index': 17, 'relative': 25.0},
        'tantalum': {'index': 18, 'relative': 35.0},
        'iron': {'index': 19, 'relative': 50.0}
    }
}


CATEGORY_LABELS = {'free':0, 'insulated':1, 'conductive':2}


def set_permittivity_type(absolute=False):
    global material_properties, PERMITTIVITY_TYPE
    if absolute:
        PERMITTIVITY_TYPE = 'absolute'
        update_relative_to_absolute()
    else:
        PERMITTIVITY_TYPE = 'relative'
    
    update_material_lookups()


def update_relative_to_absolute(epsilon_0=epsilon_0):
    global material_properties
    for _, materials in material_properties.items():
        for _, props in materials.items():
            if 'relative' in props:
                props['absolute'] = props.pop('relative') * epsilon_0


def update_material_lookups():
    global material_lookup, insulated_indices, conductive_indices, material_category_names
    
    # lookup dictionary: index => material property (name, permittivity)
    material_lookup = {
        props['index']: {
            'name': material,
            PERMITTIVITY_TYPE: props[PERMITTIVITY_TYPE],
        }
        for category in material_properties.values()
        for material, props in category.items()
    }

    # index list for insulated and conductive materials
    insulated_indices = [props['index'] for props in material_properties['insulated'].values()]
    conductive_indices = [props['index'] for props in material_properties['conductive'].values()]
    material_category_names = list(material_properties.keys())


set_permittivity_type(absolute=False)


def create_material_category_mask(grid_length=32, conductive_cell_ratio:float|None = None, conductive_cell_prob:float|None = None):
    # initialize all cells to free space (0)
    ternary_mask = np.zeros((grid_length, grid_length), dtype=int)  

    # get internel grid cell locations and count
    grid_cell_coords = [(i, j) for i in range(0, grid_length) for j in range(0, grid_length)]
    num_grid_cells = len(grid_cell_coords)

    if conductive_cell_ratio is not None and conductive_cell_prob is None:
        conductive_cell_count = max(1, int(conductive_cell_ratio * num_grid_cells))
        conductive_indices = np.random.choice(num_grid_cells, size=conductive_cell_count, replace=False)
        conductive_cells = [grid_cell_coords[idx] for idx in conductive_indices]
    elif conductive_cell_prob is not None and conductive_cell_ratio is None:
        conductive_cells = np.random.rand(grid_length, grid_length) < conductive_cell_prob
    else:
        raise ValueError("conductive_cell_ratio and conductive_cell_prob are mutually exclusive required options")

    # Apply conductive label (2) to selected conductive cells
    for row, col in conductive_cells:
        ternary_mask[row, col] = CATEGORY_LABELS['conductive']

    # Set remaining material cells to insulated (1)
    for row, col in grid_cell_coords:
        if (row, col) not in conductive_cells:
            ternary_mask[row, col] = CATEGORY_LABELS['insulated']

    binary_mask = (ternary_mask == 2).astype(int)

    return ternary_mask, binary_mask



def update_material_category_mask(initial_category_mask, connected_conductive_mask):
    # Start with an empty mask initialized as insulated by default
    ternary_mask = np.full_like(initial_category_mask, CATEGORY_LABELS['insulated'], dtype=int)
    free_mask_cond = initial_category_mask == CATEGORY_LABELS['free']
    # set all free cells in the initial category mask
    ternary_mask[free_mask_cond] = CATEGORY_LABELS['free']
    # set conductive cells backed on mask, but not overridding free cells in original mask
    ternary_mask[(connected_conductive_mask == 1) & (~free_mask_cond)] = CATEGORY_LABELS['conductive']
    dirichlet_boundary_conditions(ternary_mask, CATEGORY_LABELS['free'])
    return ternary_mask


# @todo add probabilties to this
def generate_material_index_map(ternary_mask, conductive_material_range:tuple[int,int]|None=None, conductive_material_count:int|None=None):
    # initialize all cells to 0 (free space)
    material_map = np.zeros_like(ternary_mask, dtype=int)

    insulated_mask = ternary_mask == CATEGORY_LABELS['insulated']
    conductive_mask = ternary_mask == CATEGORY_LABELS['conductive']

    # assign mask value == 1 to random insulated material index 
    material_map[insulated_mask] = np.random.choice(insulated_indices, size=insulated_mask.sum())

    if conductive_material_range is not None and conductive_material_count is None:
        # randomly selects number of materials
        min_count, max_count = conductive_material_range
        num_conductive_materials = np.random.randint(min_count, max_count + 1) if min_count != max_count else max_count
    elif conductive_material_count is not None and conductive_material_range is None:
        num_conductive_materials = conductive_material_count
    else:
        raise ValueError("Conductive material range and conductive material count are mutually exclusive required options")
    
    # randomly selects how many conductive materials are applied
    selected_conductive_indices = np.random.choice(conductive_indices, num_conductive_materials, replace=False)
    #selected_conductive_names = [material_lookup[idx][PERMITTIVITY_TYPE] for idx in selected_conductive_indices]

    # assign mask value == 2 to random conductive material index 
    material_map[conductive_mask] = np.random.choice(selected_conductive_indices, size=conductive_mask.sum())

    return material_map



def generate_permittivity_value_map(material_map):
    permittivity_value_map = np.zeros_like(material_map, dtype=float)

    unique_indices = np.unique(material_map)
    for idx in unique_indices:
        permittivity_value_map[material_map == idx] = material_lookup[idx][PERMITTIVITY_TYPE]

    return permittivity_value_map


def set_random_voltage(potential_map, binary_mask, voltage_range=(50, 200)):
    conductive_cells = np.argwhere(binary_mask == True)

    for cell in conductive_cells:
        voltage_value = np.random.uniform(voltage_range[0], voltage_range[1])
        potential_map[cell[0], cell[1]] = round(voltage_value)

    return potential_map

# https://en.wikipedia.org/wiki/Neumann_boundary_condition
def neumann_boundary_conditions(potential_map):
    potential_map[0, :] = potential_map[1, :] 
    potential_map[-1, :] = potential_map[-2, :]  
    potential_map[:, 0] = potential_map[:, 1]  
    potential_map[:, -1] = potential_map[:, -2]  
    return potential_map

# https://en.wikipedia.org/wiki/Dirichlet_problem
def dirichlet_boundary_conditions(potential_map, fixed_value=material_lookup[0][PERMITTIVITY_TYPE]):
    potential_map[0, :] = fixed_value   
    potential_map[-1, :] = fixed_value   
    potential_map[:, 0] = fixed_value  
    potential_map[:, -1] = fixed_value
    return potential_map



def mixed_boundary_conditions(potential_map, fixed_sides=(False, False, True, True), fixed_value=material_lookup[0][PERMITTIVITY_TYPE]):
    if all(fixed_sides) or all(x == False for x in fixed_sides):
        logger.error("Mixed boundary conditions requires at least one, but all sides to be fixed")
    potential_map[0, :] = fixed_value if fixed_sides[0] else potential_map[1, :] 
    potential_map[-1, :] = fixed_value if fixed_sides[1] else potential_map[-2, :]  
    potential_map[:, 0] = fixed_value if fixed_sides[2] else potential_map[:, 1]  
    potential_map[:, -1] = fixed_value if fixed_sides[3] else potential_map[:, -2]  
    return potential_map



def generate_initial_potential_map(conductive_mask, voltage_range: tuple[int, int] = None):
    initial_map = np.zeros_like(conductive_mask, dtype=float)

    if voltage_range is not None:
        voltage_map = set_random_voltage(initial_map, conductive_mask, voltage_range) 
        potential_map = neumann_boundary_conditions(voltage_map)
    else:
        potential_map = dirichlet_boundary_conditions(initial_map)

    return potential_map