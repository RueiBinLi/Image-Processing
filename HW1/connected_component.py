import cv2
import numpy as np
import os

"""
TODO Binary transfer
"""
def to_binary(img):
    if img is None:
        print('Can not load the image.')
    else:
        grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        threshold_value = 127
        _, binary_image = cv2.threshold(grayscale_image, threshold_value, 255, cv2.THRESH_BINARY_INV)

        return binary_image
        
def find(parent, i):
    if parent[i] == i:
        return i
    parent[i] = find(parent, parent[i]) # path compression
    return parent[i]

def union(parent, i, j):
    root_i = find(parent, i)
    root_j = find(parent, j)
    if root_i != root_j:
        parent[root_j] = root_i

"""
TODO Two-pass algorithm
"""
def two_pass(binary_image, connectivity):
    """
    2-pass CCA for 4 and 8 connectivity

    input:
        binary_image (np.array): A 2D NumPy array
        connectivivity (int): 4 or 8

    output:
        np.array: The labeled image
    """
    rows, cols = binary_image.shape
    labeled_image = np.zeros((rows, cols), dtype=np.int32)
    next_label = 1
    parent = [0]

    # First Pass
    for y in range(rows):
        for x in range(cols):
            if binary_image[y, x] == 255:
                neighbors = []
                
                if connectivity == 8:
                    # top-left
                    if y > 0 and x > 0 and labeled_image[y-1, x-1] > 0:
                        neighbors.append(labeled_image[y-1, x-1])
                    # top-right
                    if y > 0 and x < cols - 1 and labeled_image[y-1, x+1] > 0:
                        neighbors.append(labeled_image[y-1, x+1])
                
                # top
                if y > 0 and labeled_image[y-1, x] > 0:
                    neighbors.append(labeled_image[y-1, x])
                # left
                if x > 0 and labeled_image[y, x-1] > 0:
                    neighbors.append(labeled_image[y, x-1])

                if not neighbors:
                    labeled_image[y, x] = next_label
                    parent.append(next_label)
                    next_label += 1
                else:
                    neighbor_roots = [find(parent, label) for label in neighbors]

                    min_label = min(neighbor_roots)
                    labeled_image[y, x] = min_label

                    for label in neighbors:
                        union(parent, min_label, label)
    
    # Second pass
    for y in range(rows):
        for x in range(cols):
            if labeled_image[y, x] > 0:
                original_label = labeled_image[y, x]
                final_label = find(parent, original_label)
                labeled_image[y, x] = final_label
    
    return labeled_image


"""
TODO Seed filling algorithm
"""
def seed_filling(binary_image, connectivity):
    """
    seed_filling for 4 and 8 connectivity

    input:
        binary_image (np.array): A 2D NumPy array
        connectivity (int): 4 or 8

    output:
        np.array: The labeled image
    """
    rows, cols = binary_image.shape
    labeled_image = np.zeros((rows, cols), dtype=np.int32)
    next_label = 1

    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for y in range(rows):
        for x in range(cols):
            if binary_image[y, x] == 255 and labeled_image[y, x] == 0:
                queue = [(y, x)]
                labeled_image[y, x] = next_label

                while queue:
                    py, px = queue.pop(0)

                    for dy, dx in neighbors:
                        ny, nx = py + dy, px + dx
                    
                        if 0 <= ny < rows and 0 <= nx < cols and binary_image[ny, nx] == 255 and labeled_image[ny, nx] == 0:
                            labeled_image[ny, nx] = next_label
                            queue.append((ny, nx))

                next_label += 1

    return labeled_image

    

"""
Bonus
"""
def other_cca_algorithm(binary_image, connectivity, tile_size=(64, 64)):
    """
    Block-based CCA. Use Divide & Conquer and 2-pass to solve the CCA problem

    Input:
        binary_image (np.array): A 2D NumPy array
        connectivity (int): 4 or 8
        tile_size (int, int): The block size

    Output:
        np.array: A labeled image
    """
    rows, cols = binary_image.shape
    t_rows, t_cols = tile_size

    # 1. Divide
    num_tiles_y = (rows + t_rows - 1) // t_rows
    num_tiles_x = (cols + t_cols - 1) // t_cols

    labeled_tiles = {}
    label_offset = 0

    # 2. Conquer
    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            y_start, x_start = i * t_rows, j * t_cols
            tile = binary_image[y_start : y_start + t_rows, x_start : x_start + t_cols]

            labeled_tile= two_pass(tile, connectivity)

            non_zero_mask = labeled_tile > 0
            if np.any(non_zero_mask):
                labeled_tile[non_zero_mask] += label_offset
                label_offset = np.max(labeled_tile)

            labeled_tiles[(i, j)] = labeled_tile

    # 3. Merge
    parent = list(range(label_offset + 1)) # Initialize parent array for Union-Find

    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            tile = labeled_tiles[(i, j)]

            # Merge with the tile to the right
            if j < num_tiles_x - 1:
                right_tile = labeled_tiles[(i, j + 1)]
                for y_idx in range(min(tile.shape[0], right_tile.shape[0])):
                    if tile[y_idx, -1] > 0 and right_tile[y_idx, 0] > 0:
                        union(parent, tile[y_idx, -1], right_tile[y_idx, 0])

            # Merge with the tile below
            if i < num_tiles_y - 1:
                bottom_tile = labeled_tiles[(i + 1, j)]
                for x_idx in range(min(tile.shape[1], bottom_tile.shape[1])):
                    if tile[-1, x_idx] > 0 and bottom_tile[0, x_idx] > 0:
                        union(parent, tile[-1, x_idx], bottom_tile[0, x_idx])

    # 4. Final Relabeling
    final_labeled_image = np.zeros((rows, cols), dtype=np.int32)
    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            y_start, x_start = i * t_rows, j * t_cols
            tile = labeled_tiles[(i, j)]
            
            h, w = tile.shape
            
            # Find the root for each label in the tile
            unique_labels = np.unique(tile[tile > 0])
            for label in unique_labels:
                root = find(parent, label)
                tile[tile == label] = root

            final_labeled_image[y_start : y_start + h, x_start : x_start + w] = tile

    return final_labeled_image



"""
TODO Color mapping
"""
def color_mapping(labeled_image):
    """
    color mapping

    input: 
        labeled_image (np.array): the image which is labeled

    output:
        np.array: a 3d NumPy array representing the color-mapped image
    """
    unique_labels = np.unique(labeled_image)
    unique_labels = unique_labels[unique_labels != 0]

    if len(unique_labels) == 0:
        return np.zeros((*labeled_image.shape, 3), dtype=np.uint8)
    
    color_image = np.zeros((*labeled_image.shape, 3), dtype=np.uint8)
    num_labels = len(unique_labels)

    hue_values = np.linspace(0, 179, num_labels)

    hsv_palette = np.zeros((num_labels, 1, 3), dtype=np.uint8)
    hsv_palette[:, 0, 0] = hue_values
    hsv_palette[:, 0, 1] = 255
    hsv_palette[:, 0, 2] = 255

    bgr_palette = cv2.cvtColor(hsv_palette, cv2.COLOR_HSV2BGR)

    label_to_color = {label: bgr_palette[i][0].tolist()
                      for i, label in enumerate(unique_labels)}
    
    for label, color in label_to_color.items():
        color_image[labeled_image == label] = color

    return color_image


"""
Main function
"""
def main():

    os.makedirs("result/connected_component/two_pass", exist_ok=True)
    os.makedirs("result/connected_component/seed_filling", exist_ok=True)
    os.makedirs("result/connected_component/block_based", exist_ok=True)
    connectivity_type = [4, 8]

    for i in range(2):
        img = cv2.imread("data/connected_component/input{}.png".format(i + 1))

        for connectivity in connectivity_type:

            # TODO Part1: Transfer to binary image
            binary_img = to_binary(img)

            # TODO Part2: CCA algorithm
            two_pass_label = two_pass(binary_img, connectivity)
            seed_filling_label = seed_filling(binary_img, connectivity)
            block_based_label = other_cca_algorithm(binary_img, connectivity, (64, 64))
        
            # TODO Part3: Color mapping       
            two_pass_color = color_mapping(two_pass_label)
            seed_filling_color = color_mapping(seed_filling_label)
            block_based_color = color_mapping(block_based_label)

            cv2.imwrite("result/connected_component/two_pass/input{}_c{}.png".format(i + 1, connectivity), two_pass_color)
            cv2.imwrite("result/connected_component/seed_filling/input{}_c{}.png".format(i + 1, connectivity), seed_filling_color)
            cv2.imwrite("result/connected_component/block_based/input{}_c{}.png".format(i + 1, connectivity), block_based_color)

if __name__ == "__main__":
    main()