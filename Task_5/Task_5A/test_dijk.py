import heapq
import random


def dijkstras(graph, start, target):
    # Initialize distances to all nodes as infinity
    distances = {node: float("inf") for node in graph}
    distances[start] = 0  # Distance from start node to itself is 0

    # Priority queue to hold nodes to visit
    priority_queue = [(0, start)]  # (distance, node)

    # Dictionary to keep track of the shortest path
    previous_nodes = {}

    while priority_queue:
        # Pop the node with the smallest distance
        current_distance, current_node = heapq.heappop(priority_queue)

        # If we've reached the target node, we can reconstruct the path and return it
        if current_node == target:
            path = []
            while current_node in previous_nodes:
                path.insert(0, current_node)
                current_node = previous_nodes[current_node]
            path.insert(0, start)
            return path

        # Check distances to neighbors of the current node
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight

            # If a shorter path is found, update distance and previous node
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    # If no path is found
    return []


graph = {
    "P0": {"P1": 27},
    "P1": {"P0": 27, "P2": 97, "P5": 50, "A": 35},
    "P2": {"P4": 52, "P1": 97, "A": 62},
    "P3": {"P4": 103, "P2": 160, "B": 45},
    "P4": {"P2": 52, "P7": 51, "P5": 100},
    "P5": {"P6": 57, "P1": 50, "P4": 100},
    "P6": {"P7": 100, "P11": 40, "P5": 57, "D": 38},
    "P7": {"P6": 100, "P4": 51, "D": 62, "P8": 103, "P10": 38, "C": 65},
    "P8": {"P7": 103, "C": 41, "P3": 50, "P9": 38},
    "P9": {"P8": 38, "P10": 102},
    "P10": {"P9": 102, "P11": 101},
    "P11": {"P10": 101, "E": 71, "P6": 40},
    "A": {"P1": 35, "P2": 62},
    "B": {"P3": 45, "P4": 60},
    "C": {"P7": 65, "P8": 41},
    "D": {"P7": 62, "P6": 38},
    "E": {"P11": 71},
}


start_node = "P0"
node_priority = input("Enter node priority (e.g., ABCDE): ")

# Initialize the path with the start node
path = [start_node]

# Traverse from each priority node to the next
for i in range(len(node_priority)):
    current_node = path[-1]  # Get the last node in the path
    next_node = node_priority[i]
    shortest_path = dijkstras(graph, current_node, next_node)
    path.extend(
        shortest_path[1:]
    )  # Extend the path, excluding the first node (already in the path)

# Return to P0 from the last priority node
shortest_path_to_P0 = dijkstras(graph, path[-1], start_node)
path.extend(
    shortest_path_to_P0[1:]
)  # Extend the path, excluding the first node (already in the path)

# Construct the path string
path_str = "".join(path)+'s'
# for i in range(0, len(path_str)):
#     if path_str[i] == "A":
#         path_str = path_str[:i] + "Az" + path_str[i + 1 :]
#         if path_str[i - 1] == "2":
#             path_str = path_str[:i] + "a" + path_str[i + 1 :]
#     elif path_str[i] == "B":
#         path_str = path_str[:i] + "Bz" + path_str[i + 1 :]
#         if path_str[i - 1] == "3":
#             path_str = path_str[:i] + "b" + path_str[i + 1 :]
#     elif path_str[i] == "C":
#         path_str = path_str[:i] + "Cz" + path_str[i + 1 :]
#         if path_str[i - 1] == "8":
#             path_str = path_str[:i] + "c" + path_str[i + 1 :]
#     elif path_str[i] == "D":
#         path_str = path_str[:i] + "Dz" + path_str[i + 1 :]
#         if path_str[i - 1] == "7":
#             path_str = path_str[:i] + "d" + path_str[i + 1 :]

print("Shortest Path:", path_str)
