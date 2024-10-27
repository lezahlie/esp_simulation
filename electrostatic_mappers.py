from setup_logger import setup_logger
logger = setup_logger(__file__, log_stderr=True, log_stdout=True)
from utilities import np

material_properties = {
    'free': {
        'air_vacuum': {'index': 0, 'relative': 1.0}
    },
    'insulated': {
        'teflon': {'index': 1, 'relative': 2.1},
        'polyethylene': {'index': 2, 'relative': 2.25},
        'nylon': {'index': 3, 'relative': 3.4},
        'epoxy_resin': {'index': 4, 'relative': 3.5},
        'FR4': {'index': 5, 'relative': 4.5},
        'glass': {'index': 6, 'relative': 5.0},
        'rubber': {'index': 7, 'relative': 7.0}
    },
    'conductive': {
        'aluminum': {'index': 8, 'relative': 1.5},
        'nickel': {'index': 9, 'relative': 1.8},
        'stainless_steel': {'index': 10, 'relative': 2.0},
        'bronze': {'index': 11, 'relative': 2.5},
        'copper_alloy': {'index': 12, 'relative': 2.8},
        'zinc': {'index': 13, 'relative': 3.0},
        'tin': {'index': 14, 'relative': 5.0},
        'lead': {'index': 15, 'relative': 6.0},
        'graphite': {'index': 16, 'relative': 10.0},
        'silicon': {'index': 17, 'relative': 11.7},
        'tantalum': {'index': 18, 'relative': 26.0},
        'iron': {'index': 19, 'relative': 50.0}
    }
}


def set_permittivity_type(absolute=False):
    global material_properties, PERMITTIVITY_TYPE
    if absolute:
        PERMITTIVITY_TYPE = 'absolute'
        update_relative_to_absolute()
    else:
        PERMITTIVITY_TYPE = 'relative'
    
    update_material_lookups()


def update_relative_to_absolute(epsilon_0=8.854e-12):
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
            PERMITTIVITY_TYPE: props[PERMITTIVITY_TYPE],  # Use the correct permittivity type
        }
        for category in material_properties.values()
        for material, props in category.items()
    }

    # index list for insulated and conductive materials
    insulated_indices = [props['index'] for props in material_properties['insulated'].values()]
    conductive_indices = [props['index'] for props in material_properties['conductive'].values()]
    material_category_names = list(material_properties.keys())


set_permittivity_type(absolute=False)


def create_material_location_masks(grid_length=32, material_ratio=1.0, conductive_ratio=0.25):
    # initialize all cells to free space (0)
    ternary_mask = np.zeros((grid_length, grid_length), dtype=int)  
    
    # get internel grid cell locations and count
    internal_grid_cells = [(i, j) for i in range(1, grid_length - 1) for j in range(1, grid_length - 1)]
    total_internal_cells = len(internal_grid_cells)

    # calculate total material cells based on material_ratio and the total conductive materials
    material_cell_count = int(material_ratio * total_internal_cells)
    conductive_cell_count = int(conductive_ratio * material_cell_count)

    # rangomly select indices for material cells
    material_indices = np.random.choice(len(internal_grid_cells), size=material_cell_count, replace=False)

    # set conductive cells to 2
    conductive_indices = np.random.choice(material_indices, size=conductive_cell_count, replace=False)
    for idx in conductive_indices:
        row, col = internal_grid_cells[idx]
        ternary_mask[row, col] = 2

    # set remaining material cells to 1 (insulated)
    insulated_indices = [idx for idx in material_indices if idx not in conductive_indices]
    for idx in insulated_indices:
        row, col = internal_grid_cells[idx]
        ternary_mask[row, col] = 1

    binary_mask = (ternary_mask == 2).astype(bool)

    return ternary_mask, binary_mask


def generate_material_index_map(ternary_mask):
    # initialize all cells to 0 (free space)
    material_map = np.zeros_like(ternary_mask, dtype=int)
    
    # assign mask value == 1 to random insulated material index 
    material_map[ternary_mask == 1] = np.random.choice(insulated_indices, size=(ternary_mask == 1).sum())

    # assign mask value == 2 to random conductive material index 
    material_map[ternary_mask == 2] = np.random.choice(conductive_indices, size=(ternary_mask == 2).sum())

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


def neumann_boundary_conditions(potential_map):
    potential_map[0, :] = potential_map[1, :] 
    potential_map[-1, :] = potential_map[-2, :]  
    potential_map[:, 0] = potential_map[:, 1]  
    potential_map[:, -1] = potential_map[:, -2]  
    return potential_map


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