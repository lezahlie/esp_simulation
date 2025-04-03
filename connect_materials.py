from setup_logger import setup_logger
logger = setup_logger(__file__, log_stdout=True, log_stderr=True)
from utilities import np, ndimg, deque

N=(-1, 0)
W=(0, -1)
E=(0, 1)
S=(1, 0)

FOUR_NEIGHBORHOOD = [N, S, W, E]
FOUR_KERNEL = np.array([[0, 1, 0], [1, 1, 1],[0, 1, 0]])
EIGHT_KERNEL = np.array([[1, 1, 1],[1, 0, 1],[1, 1, 1]])


# retains the orginal border cells
def preserve_borders(A, B):
    A[0, B[0, :] == 1] = B[0, B[0, :] == 1]         # top
    A[-1, B[-1, :] == 1] = B[-1, B[-1, :] == 1]     # bottom
    A[B[:, 0] == 1, 0] = B[B[:, 0] == 1, 0]         # left
    A[B[:, -1] == 1, -1] = B[B[:, -1] == 1, -1]     # right
    return A


# flips all the cell states
def reverse_shape(shape):
    flipped_shape = np.logical_not(shape).astype(int)
    return flipped_shape

def manhattan_distance(point_a, point_b):
    return abs(point_a[0] - point_b[0]) + abs(point_a[1] - point_b[1])

def is_internal(pos, max_pos):
    return 0 <= pos[0] < max_pos[0] and 0 <= pos[1] < max_pos[1]


def close_to_goal(current, neighbor, goal):
    current_dist = manhattan_distance(current, goal)
    neighbor_dist = manhattan_distance(neighbor, goal)
    return neighbor_dist <= current_dist


def smooth_path(grid):
    path_mask = (grid == 1).astype(np.uint8)
    smoothed = ndimg.binary_closing(path_mask, structure=FOUR_KERNEL)
    return smoothed.astype(int)

def bfs_flood_fill(grid, start, goal):
    queue = deque([start])
    visited = {start}
    
    while queue:
        curr_xy = queue.popleft()
        grid[curr_xy] = 1

        if curr_xy == goal:
            break

        neighbors = [(curr_xy[0] + dx, curr_xy[1] + dy) 
                        for dx, dy in FOUR_NEIGHBORHOOD]
        
        for next_xy in neighbors:
            if (is_internal(next_xy, grid.shape) 
                and next_xy not in visited 
                and grid[next_xy] == 0 
                and close_to_goal(curr_xy, next_xy, goal)):
                
                visited.add(next_xy)
                queue.append(next_xy)

    smoothed_paths = smooth_path(grid)
    grid[smoothed_paths == 1] = 1


def find_closest_region(region_a, region_b):
    min_distance = float('inf')
    closest_pair = (None, None)
    for point_a in region_a:
        for point_b in region_b:
            distance = manhattan_distance(point_a, point_b)
            if distance < min_distance:
                min_distance = distance
                closest_pair = (tuple(point_a), tuple(point_b))
    return closest_pair


def get_region_labels(grid_in, ignore_borders=True):
    grid_out = grid_in.copy()
    if ignore_borders:
        grid_out[0, :] = 0
        grid_out[-1, :] = 0
        grid_out[:, 0] = 0
        grid_out[:, -1] = 0

    connection = np.array([[0, 1, 0],
                            [1, 1, 1], 
                            [0, 1, 0]])
    
    return ndimg.label(grid_out, structure=connection)


class UnionFinder:
    def __init__(self, n):
        self.parent = {i: i for i in range(1, n + 1)}
    
    def find(self, x):
        while self.parent[x] != x:
            x = self.parent[x]
        return x
    
    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)

    def count_unique(self):
        return len(set(self.find(i) for i in self.parent))

    

def connect_regions(grid):
    label_grid, num_features = get_region_labels(grid)

    if num_features <= 1:
        logger.debug("All regions are already connected")
        return grid

    regions = {lab: np.argwhere(label_grid == lab)
                for lab in range(1, num_features + 1)}
    
    edges = []
    for i in range(1, num_features + 1):
        for j in range(i + 1, num_features + 1):
            region_a = regions[i]
            region_b = regions[j]
            if region_a.size > 0 and region_b.size > 0:
                start, goal = find_closest_region(region_a, region_b)
                if start and goal:
                    distance = manhattan_distance(start, goal)
                    edges.append((distance, i, j, start, goal))

    edges.sort(key=lambda x: x[0])
    ufind = UnionFinder(num_features)

    for distance, i, j, start, goal in edges:
        if ufind.find(i) != ufind.find(j):
            ufind.union(i, j)
            bfs_flood_fill(grid, start, goal)
            if ufind.count_unique() == 1:
                break

    return grid


# one iteration of cellular automata
def cellular_automata_rules(shape_in):
    shape_out = shape_in.copy()
    neighbor_counts = ndimg.convolve(shape_out, EIGHT_KERNEL, mode='wrap')
    mask_dies = (shape_out == 1) & (neighbor_counts < 4)
    mask_born = (shape_out == 0) & (neighbor_counts > 4)

    shape_out = np.where(mask_dies, 0, shape_out)
    shape_out = np.where(mask_born, 1, shape_out)
    shape_out = preserve_borders(shape_out, shape_in)
    return shape_out


# applies CA rules until the shape stop changing or exceeds 100 iterations
def connect_cellular_automata_shape(conductive_mask, reverse=False, max_iterations=100):
    image_shape = conductive_mask.astype(int)
    prev_shape = image_shape
    new_shape = cellular_automata_rules(image_shape)

    count = 0
    while not np.array_equal(prev_shape, new_shape) and count < max_iterations:
        prev_shape = new_shape
        new_shape = cellular_automata_rules(prev_shape)
        count += 1

    # reverse shape
    unconnected_shape = np.logical_not(new_shape).astype(int) if reverse else new_shape
    # connect materials and fill region
    connected_shape = connect_regions(unconnected_shape)
    return connected_shape


def test_connect_materials(random_seed, conductive_ratio=None, conductive_prob=None):
    from electrostatic_mappers import (create_material_category_mask, 
                                        generate_material_index_map, 
                                        update_material_category_mask)

    grid_length = 16

    if random_seed:
        np.random.seed(random_seed)

    initial_category_mask, initial_conductive_mask = create_material_category_mask(grid_length, conductive_ratio, conductive_prob)
    print(f"{'-'*100}", f"SEED: {random_seed}",  f"P(Cell is conductive): {conductive_prob}", f"Conductive cell ratio: {conductive_ratio}", f"{'-'*100}", sep='\n')
    print("initial_category_mask",initial_category_mask, sep='\n')
    print("initial_conductive_mask",initial_conductive_mask, sep='\n')

    connected_conductive_mask = connect_cellular_automata_shape(initial_conductive_mask, reverse=True)
    print("\nConnected Conductive Mask:\n", connected_conductive_mask , sep='\n')

    connected_category_mask = update_material_category_mask(initial_category_mask, connected_conductive_mask)
    print("\nconnected_category_mask", connected_category_mask, sep='\n')

    connected_material_map = generate_material_index_map(connected_category_mask)
    print("\nfinal_connected_material_map", connected_material_map, sep='\n')


def main():
    max_seed = 10

    conductive_prob = 0.5
    for seed in range(1, max_seed):
        test_connect_materials(seed, conductive_prob=conductive_prob)

    conductive_ratio = 0.5
    for seed in range(1, max_seed):
        test_connect_materials(seed, conductive_ratio=conductive_ratio)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(e)