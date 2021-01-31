#!/usr/bin/env python3

from util import read_osm_data, great_circle_distance, to_local_kml_url
import time

# NO ADDITIONAL IMPORTS!


ALLOWED_HIGHWAY_TYPES = {
    'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified',
    'residential', 'living_street', 'motorway_link', 'trunk_link',
    'primary_link', 'secondary_link', 'tertiary_link',
}


DEFAULT_SPEED_LIMIT_MPH = {
    'motorway': 60,
    'trunk': 45,
    'primary': 35,
    'secondary': 30,
    'residential': 25,
    'tertiary': 25,
    'unclassified': 25,
    'living_street': 10,
    'motorway_link': 30,
    'trunk_link': 30,
    'primary_link': 30,
    'secondary_link': 30,
    'tertiary_link': 25,
}


def build_auxiliary_structures(nodes_filename, ways_filename):
    """
    Create any auxiliary structures you are interested in, by reading the data
    from the given filenames (using read_osm_data)
    """
    nodes = {}
    for way in read_osm_data(ways_filename):
        highway_type = way['tags'].get('highway', '( ͡° ͜ʖ ͡°)')
        if highway_type in ALLOWED_HIGHWAY_TYPES:
            nodes_along_way = way['nodes']  # List of nodes along this way
            for i in range(len(nodes_along_way) - 1):
                # A pair of adjacent nodes along this way
                left = nodes_along_way[i]
                right = nodes_along_way[i + 1]
                default_speed_limit = DEFAULT_SPEED_LIMIT_MPH[highway_type]
                # If this way doesn't have a speed limit tag, we use the default value based on highway type
                speed_limit = way['tags'].get('maxspeed_mph', default_speed_limit)

                def build_data(root, adjacent):
                    """
                        root: ID of some node along way
                        adjacent: ID of some node adjacent to root node along way
                    """
                    new_node_data_struct = {'adjacent': {adjacent: speed_limit}}  # Init dict for node data structure
                    root_data = nodes.get(root, new_node_data_struct)
                    # There might be another way where root and adjacent are directly adjacent, so our
                    # speed limit is the max of the speed limits of those two ways:
                    root_data['adjacent'][adjacent] = max(root_data['adjacent'].get(adjacent, 0), speed_limit)
                    nodes[root] = root_data  # Add the data on root to our dictionary of node data

                build_data(left, right)
                if not way['tags'].get('oneway', '( ͡° ͜ʖ ͡°)') == 'yes':
                    # If this isn't a oneway way, we can build the data structure for the next node as well
                    build_data(right, left)
                elif right == nodes_along_way[-1]:
                    # In non-oneway ways, the above build_data(right, left) call creates the data structure
                    # for the final node at the same time as the penultimate one. However, in the case of a
                    # oneway path, we have to do it manually:
                    nodes[right] = nodes.get(right, {'adjacent': {}})

    for node in read_osm_data(nodes_filename):
        id = node['id']
        if id in nodes:
            # If the id of this node in the generator was on a valid way, we add the data about that node
            # to its dictionary in nodes.
            # Add lat/lon data
            nodes[id]['lat'] = node['lat']
            nodes[id]['lon'] = node['lon']

    return nodes


class Heap:

    def __init__(self, prop, start=None, start_item=None):
        self.property = 'min' if (prop == 'min') else 'max'  # Heap property
        self.heap = []  # List representation of the heap
        self.items = []  # A list of the items corresponding to each index in the heap
        self.size = 0
        if isinstance(start, list):
            self.heap = start[:]
            self.items = start_item if start_item is not None else [None] * len(start)
            self.size = len(self.heap)
            for i in range(len(start) // 2, -1, -1):
                # Second half of the heap is comprised entirely of leaves, so we know it fulfills our heap property
                # We loop backwards over the array and max heapify down so we always maintain our heap property
                # at every index after i
                self.heapify_down(i)
        elif start is not None:
            self.add(start, start_item)
            self.size = 1

    def parent(self, i):
        # Returns the index of i's parent if it has one
        return (i + 1) // 2 - 1 if i > 0 else i

    def left(self, i):
        # Returns the index of i's left child if it has one
        return 2 * i + 1 if i < self.size else i

    def right(self, i):
        # Returns the index of i's right child if it has one
        return 2 * (i + 1) if i < self.size else i

    def add(self, val, item=None):
        # Add value to heap
        self.heap.append(val)
        self.items.append(item)
        self.size += 1
        self.heapify_up(self.size - 1)

    def next(self):
        # Get the value at the top of the heap
        if self.size > 0:
            if self.size == 1:
                self.size -= 1
                return self.heap.pop(0), self.items.pop(0)
            else:
                # Swap element at the top of the heap to the end
                self.swap(0, self.size - 1)
                top = self.heap.pop(self.size - 1)
                top_item = self.items.pop(self.size - 1)
                self.size -= 1
                self.heapify_down(0)  # Heapify from the top
                return top, top_item

    def heapify_up(self, i):
        # Assume everything below i fulfills the heap property, shift value at index i up until
        # our heap property is fulfilled across the entire heap
        p = self.parent(i)
        if not p == i and ((self.property == 'max' and self.heap[i] > self.heap[p]) or
                       (self.property == 'min' and self.heap[i] < self.heap[p])):
            # If node i violates this heap's heap property, swap it with its parent, then check again:
            self.swap(i, p)
            self.heapify_up(p)

    def heapify_down(self, p):
        # Assume everything below p fulfills the heap property, shift value at index p down until
        # our heap property is fulfilled across the entire heap
        l, r = self.left(p), self.right(p)
        if l >= self.size:
            # If p has no children, we do nothing
            return

        if self.property == 'max':
            c = l if r >= self.size or self.heap[l] > self.heap[r] else r
            if self.heap[p] < self.heap[c]:
                # If node p violates this heap's max heap property, swap it with its larger child, then check again:
                self.swap(p, c)
                self.heapify_down(c)
        else:
            # if property == 'min'
            c = l if r >= self.size or self.heap[l] < self.heap[r] else r
            if self.heap[p] > self.heap[c]:
                # If node p violates this heap's min heap property, swap it with its smaller child, then check again:
                self.swap(p, c)
                self.heapify_down(c)

    def swap(self, a, b):
        # Swaps the elements of heap and items at indices a and b
        self.heap[a], self.heap[b] = self.heap[b], self.heap[a]
        self.items[a], self.items[b] = self.items[b], self.items[a]

    def empty(self):
        # Returns true if this heap has no elements
        return self.size == 0

    def __str__(self):
        # Returns the heap in the form of a list
        return str(self.heap)


def find_min_cost_path(data, start, is_goal, get_children, cost, heuristic=lambda x: 0):
    """haha, uniform cost search go brrr"""
    paths = Heap('min', heuristic(start), ([start], 0))  # Min heap of paths and their respective costs (sorted by heuristic cost)
    seen = set()  # Set of nodes we've already found shorter paths to

    #  T H E  S E A R C H  L O O P  B E G I N S
    while not paths.empty():
        next_path = paths.next()  # get the minimum cost path (heuristic cost, (path, path cost))
        min_cost_path = next_path[1][0]
        min_cost = next_path[1][1]
        terminal_node = min_cost_path[-1]

        while terminal_node in seen:
            # If we've already found a path to the same node with a lower cost, we pick a new next_path
            if paths.empty():
                # If we run out of paths to search, we return nothing
                return None
            next_path = paths.next()
            min_cost_path = next_path[1][0]
            min_cost = next_path[1][1]
            terminal_node = min_cost_path[-1]

        if is_goal(terminal_node):
            return min_cost_path

        seen.add(terminal_node)
        children = get_children(terminal_node)
        for c in children:
            if c not in seen:
                # If this child does not have an existing path to it already, we build a
                # data structure for it and at it to our min heap
                path_to_c = min_cost_path + [c]
                c_cost = min_cost + cost(data, terminal_node, c)
                c_heuristic = c_cost + heuristic(c)
                paths.add(c_heuristic, (path_to_c, c_cost))
    #  T H E  S E A R C H  L O O P  E N D S

    return None  # We failed to find a path to the goal node. Very sad. Return nothing :(


def get_dist_cost(data, start_node_id, end_node_id):
    """
    Calculates the cost of the direct path (which is assume to exist) between
    specified start and end nodes based on the distance between them.

    Parameters:
        data: The auxiliary data structure (a dictionary) that stores information
        about nodes and the ways that connect them
        start_node_id: The integer id of the start node in data
        end_node_id: The integer id of the end node in data
    """
    p1 = get_coords(data, start_node_id)
    p2 = get_coords(data, end_node_id)
    return great_circle_distance(p1, p2)


def get_coords(data, id):
    """
    Returns the GPS coordinates of a node in the form of a (lat, lon) tuple given its id number
    """
    return data[id]['lat'], data[id]['lon']


def find_short_path_nodes(aux_structures, node1, node2):
    """
    Return the shortest path between the two nodes

    Parameters:
        aux_structures: the result of calling build_auxiliary_structures
        node1: node representing the start location
        node2: node representing the end location

    Returns:
        a list of node IDs representing the shortest path (in terms of
        distance) from node1 to node2
    """
    p = find_min_cost_path(
        aux_structures,
        node1,
        lambda x: x == node2,
        lambda parent_id: aux_structures[parent_id]['adjacent'],
        get_dist_cost,
        lambda x: gcd_heuristic(aux_structures, x, node2))
    return list(p) if p is not None else None


def gcd_heuristic(data, node1, node2):
    return great_circle_distance(get_coords(data, node1), get_coords(data, node2))


def find_short_path(aux_structures, loc1, loc2):
    """
    Return the shortest path between the two locations

    Parameters:
        aux_structures: the result of calling build_auxiliary_structures
        loc1: tuple of 2 floats: (latitude, longitude), representing the start
              location
        loc2: tuple of 2 floats: (latitude, longitude), representing the end
              location

    Returns:
        a list of (latitude, longitude) tuples representing the shortest path
        (in terms of distance) from loc1 to loc2.
    """
    node1 = get_closest_node(aux_structures, loc1)
    node2 = get_closest_node(aux_structures, loc2)
    p = find_min_cost_path(
        aux_structures,
        node1,
        lambda x: x == node2,
        lambda parent_id: aux_structures[parent_id]['adjacent'],
        get_dist_cost,
        lambda x: gcd_heuristic(aux_structures, x, node2))
    return get_coord_list(aux_structures, p) if p is not None else None


def get_closest_node(data, loc):
    """
    Calculates the closest node in the given dataset to a specified query location

    Parameters:
        data: The auxiliary data structure (a dictionary) that stores information
        about nodes and the ways that connect them
        loc: The query location, given in terms of a tuple of two floats (latitude, longitude)
    """
    min_dist = None
    closest = None
    for i in data:
        # Standard min-value search loop
        dist = great_circle_distance(get_coords(data, i), loc)
        if closest is None or dist < min_dist:
            closest = i
            min_dist = dist
    return closest


def get_coord_list(data, ids):
    """ Converts a list of node ids to (latitude, longitude) tuples """
    l = len(ids)
    coord_list = [None] * l
    for i in range(l):
        coord_list[i] = get_coords(data, ids[i])
    return coord_list


def find_fast_path(aux_structures, loc1, loc2):
    """
    Return the shortest path between the two locations, in terms of expected
    time (taking into account speed limits).

    Parameters:
        aux_structures: the result of calling build_auxiliary_structures
        loc1: tuple of 2 floats: (latitude, longitude), representing the start
              location
        loc2: tuple of 2 floats: (latitude, longitude), representing the end
              location

    Returns:
        a list of (latitude, longitude) tuples representing the shortest path
        (in terms of time) from loc1 to loc2.
    """
    node1 = get_closest_node(aux_structures, loc1)
    node2 = get_closest_node(aux_structures, loc2)
    p = find_min_cost_path(
        aux_structures,
        node1,
        lambda x: x == node2,
        lambda parent_id: aux_structures[parent_id]['adjacent'],
        get_speed_cost)
    return get_coord_list(aux_structures, p) if p is not None else None


def get_speed_cost(data, start_node_id, end_node_id):
    """
    Calculates the cost of the direct path (which is assume to exist) between
    specified start and end nodes based on the speed limit between them.

    Parameters:
        data: The auxiliary data structure (a dictionary) that stores information
        about nodes and the ways that connect them
        start_node_id: The integer id of the start node in data
        end_node_id: The integer id of the end node in data
    """
    start_node = data[start_node_id]
    dist_cost = get_dist_cost(data, start_node_id, end_node_id)
    # Get the speed limit along the way that connects the starting and ending nodes
    speed_limit_between_nodes = start_node['adjacent'][end_node_id]
    return dist_cost / speed_limit_between_nodes  # Cost = Time = Distance/Rate


def print_data(data):
    print('Nodes:')
    for i in data:
        print("id: " + str(i) + " | " + str(data[i]))


if __name__ == '__main__':
    # additional code here will be run only when lab_old.py is invoked directly
    # (not when imported from test.py), so this is a good place to put code
    # used, for example, to generate the results for the online questions.

    # mit_set = build_auxiliary_structures('resources/mit.nodes', 'resources/mit.ways')
    # midwest_set = build_auxiliary_structures('resources/midwest.nodes', 'resources/midwest.ways')
    # cambridge_set = build_auxiliary_structures('resources/cambridge.nodes', 'resources/cambridge.ways')

    # print(len(find_fast_path(midwest_set, (41.375288, -89.459541), (41.452802, -89.443683))))
    # Without heuristic: 372625 pulls
    # With heuristic: 45928 pulls

    # Start Server: python3 server.py cambridge
    # Distance Path: http://localhost:6009/
    # Fast Path: http://localhost:6009/?type=fast
    pass





