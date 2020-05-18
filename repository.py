#Data Structure Repository
#Contents: Node, LinkedList, HashMap, Tree, MinHeap, Graph (Directed, Weighted), BubbleSort, MergeSort, Quicksort, RadixSort, LinearSearch, BinarySearch, DepthFirstSearch, Dijkstra's Algorithm

#Node (4/30/2020)
class Node:
    def __init__(self, value, link_node = None):
        self.value = value
        self.link_node = link_node
        
    def get_value(self):
        return self.value
        
    def get_link_node(self):
        return self.link_node
        
    def set_link_node(self, new_link):
        self.link_node = new_link

#Linked List (4/30/2020)
class LinkedList:
    def __init__(self, head_node = None):
        self.head_node = head_node
    
    def get_head_node(self):
        return self.head_node
        
    def insert(self, new_node):
        current_node = self.get_head_node()
        
        if not current_node:
            self.head_node = new_node
            
        while current_node:
            next_node = current_node.get_link_node()
            if not next_node:
                current_node.set_link_node(new_node)
            current_node = next_node
            
    def __iter__(self):
        current_node = self.get_head_node()
        while current_node:
            yield current_node.get_value()
            current_node = current_node.get_link_node()
            
#Hash Map with Separate Chaining (4/30/2020)
#Python has built-in HashMaps called Dictionaries
class HashMap:
  def __init__(self, size):
    self.array_size = size
    self.array = [LinkedList() for i in range(size)]

  def hash(self, key):
    return sum(key.encode())

  def compress(self, hash_code):
    return hash_code % self.array_size

  def assign(self, key, value):
    array_index = self.compress(self.hash(key))
    payload = Node([key, value])
    list_at_array = self.array[array_index]
    for item in list_at_array:
      if item[0] == key:
        item[1] = value
    list_at_array.insert(payload)

  def retrieve(self, key):
    array_index = self.compress(self.hash(key))
    list_at_index = self.array[array_index]
    for item in list_at_index:
      if item[0] == key:
        return item[1]
    return None
    
#Tree (4/30/2020)
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []
        
    def add_child(self, child_node):
        self.children.append(child_node)
        
    def remove_child(self, child_node):
        self.children = [child for child in self.children if child != child_node]
        
    def traverse(self):
        nodes_to_visit = [self]
        while len(nodes_to_visit) > 0:
            current_node = nodes_to_visit.pop()
            print(current_node.value)
            nodes_to_visit += current_node.children
            
#Min Heap (5/1/2020)
class MinHeap:
    def __init__(self):
        self.heap_list = [None]
        self.count = 0
        
    def parent_idx(self, idx):
        return idx // 2
        
    def left_child_idx(self, idx):
        return idx * 2
        
    def right_child_idx(self, idx):
        return idx * 2 + 1
        
    def child_present(self, idx):
        return self.left_child_idx(idx) <= self.count
    
    def retrieve_min(self):
        if self.count == 0:
            print("No items in heap")
            return None
        min = self.heap_list[1]
        self.heap_list[1] = self.heap_list[self.count]
        self.count -= 1
        self.heap_list.pop()
        self.heapify_down()
        return min
        
    def add(self, element):
        self.count += 1
        self.heap_list.append(element)
        self.heapify_up()
        
    def get_smaller_child_idx(self, idx):
        if self.right_child_idx(idx) > self.count:
            return self.left_child_idx(idx)
        else:
            left_child = self.heap_list[self.left_child_idx(idx)]
            right_child = self.heap_list[self.right_child_idx(idx)]
            if left_child < right_child:
                return self.left_child_idx(idx)
            else:
                return self.right_child_idx(idx)
                
    def heapify_up(self):
        idx = self.count
        while self.parent_idx(idx) > 0:
            child = self.heap_list[idx]
            parent = self.heap_list[self.parent_idx(idx)]
            if parent > child:
                self.heap_list[idx] = parent
                self.heap_list[self.parent_idx(idx)] = child
            idx = self.parent_idx(idx)
            
    def heapify_down(self):
        idx = 1
        while self.child_present(idx):
            smaller_child_idx = self.get_smaller_child_idx(idx)
            child = self.heap_list[smaller_child_idx]
            parent = self.heap_list[idx]
            if parent > child:
                self.heap_list[idx] = child
                self.heap_list[smaller_child_idx] = parent
            idx = smaller_child_idx
            
#Directed Graph with Weights (5/3/2020)
class Vertex:
    def __init__(self, value):
        self.value = value
        self.edges = {}
        
    def add_edge(self, vertex, weight = 0):
        self.edges[vertex] = weight
        
    def get_edges(self):
        return list(self.edges.keys())
        
class Graph:
    def __init__(self, directed = False):
        self.graph_dict = {}
        self.directed = directed
    
    def add_vertex(self, vertex):
        self.graph_dict[vertex.value] = vertex
        
    def add_edge(self, from_vertex, to_vertex, weight = 0):
        self.graph_dict[from_vertex.value].add_edge(to_vertex.value, weight)
        if not self.directed:
            self.graph_dict[to_vertex.value].add_edge(from_vertex.value, weight)
            
    def find_path(self, start_vertex, end_vertex):
        start = [start_vertex]
        seen = {}
        while len(start) > 0:
            current_vertex = start.pop(0)
            seen[current_vertex] = True
            if current_vertex == end_vertex:
                return True
            else:
                vertex = self.graph_dict[current_vertex]
                next_vertices = vertex.get_edges()
                next_vertices = [vertex for vertex in next_vertices if vertex not in seen]
                start.extend(next_vertices)
        return False
            
#BubbleSort (5/5/2020)
def swap(arr, index_1, index_2):
    temp = arr[index_1]
    arr[index_1] = arr[index_2]
    arr[index_2] = temp
    
def bubble_sort(arr):
    for el in arr:
        for index in range(len(arr)-1):
            if arr[index] > arr[index + 1]:
                swap(arr, index, index+1)

#MergeSort (5/6/2020)
def merge_sort(items):
    if len(items) <= 1:
        return items
    
    middle_index = len(items) // 2
    left_split = items[:middle_index]
    right_split = items[middle_index:]
    
    left_sorted = merge_sort(left_split)
    right_sorted = merge_sort(right_split)
    
    return merge(left_sorted, right_sorted)
    
def merge(left, right):
    result = []
    
    while left and right:
        if left[0] < right[0]:
            result.append(left[0])
            left.pop(0)
        else:
            result.append(right[0])
            right.pop(0)
            
    if left:
        result += left
    if right:
        result += right
        
    return result
        
#Quicksort (in-place implementation) (5/7/2020)
from random import randrange

def quicksort(list, start, end):
    if start >= end:
        return
    
    pivot_idx = randrange(start, end + 1)
    pivot_element = list[pivot_idx]
    
    list[end], list[pivot_idx] = list[pivot_idx], list[end]
    
    less_than_pointer = start
    
    for i in range(start, end):
        if list[i] < pivot_element:
            list[i], list[less_than_pointer] = list[less_than_pointer], list[i]
            less_than_pointer += 1
    list[end], list[less_than_pointer] = list[less_than_pointer], list[end]
    
    quicksort(list, start, less_than_pointer - 1)
    quicksort(list, less_than_pointer + 1, end)
    
#RadixSort (least significant digit) (5/8/2020)
def radix_sort(to_be_sorted):
    maximum_value = max(to_be_sorted)
    max_exponent = len(str(maximum_value))
    being_sorted = to_be_sorted[:]
    
    for exponent in range(max_exponent):
        position = exponent + 1
        index = -position
        
        digits = [[] for i range(10)]
        
        for number in being_sorted:
            number_as_a_string = str(number)
            try:
                digit = number_as_a_string[index]
                digit = int(digit)
            except IndexError:
                digit = 0
                
            digits[digit].append(number)
            
        being_sorted = []
        for numeral in digits:
            being_sorted.extend(numeral)
            
    return being_sorted
    
#LinearSearch (5/12/2020)
def linear_search(search_list, target):
    for el in range(len(search_list)):
        if search_list[el] == target:
            return el
    raise ValueError("{} not in list".format(target))
    
#LinearSearch check for multiple occurrences (5/12/2020)
def linear_mult(search_list, target):
    matches = []
    for el in range(len(search_list)):
        if search_list[el] == target:
            matches.append(el)
    if matches:
        return matches
    else:
        raise ValueError("{} not in list".format(target))
 
#BinarySearch (5/12/2020)
def binary_search(sorted_list, target):
    if not sorted_list:
        return "value not found"
    mid_idx = len(sorted_list) // 2
    mid_val = sorted_list[mid_idx]
    if mid_val == target:
        return mid_idx
    if mid_val > target:
        left_half = sorted_list[0:mid_idx]
        return binary_search(left_half, target)
    if mid_val < target:
        right_half = sorted_list[mid_idx+1:]
        result = binary_search(right_half, target)
        if result == "value not found":
            return result
        else:
            return result + mid_idx + 1
            
#BinarySearch with pointers (better implementation) (5/12/2020)
def binary_search2(sorted_list, left_pointer, right_pointer, target):
    if left_pointer >= right_pointer:
        return "value not found"
        
    mid_idx = (left_pointer + right_pointer) // 2
    mid_val = sorted_list[mid_idx]
    
    if mid_val == target:
        return mid_idx
    if mid_val > target:
        return binary_search2(sorted_list, left_pointer, mid_idx, target)
    if mid_val < target:
        return binary_search2(sorted_list, mid_idx + 1, right_pointer, target)
        
#Iterative BinarySearch (5/13/2020)
#Works with sparsely sorted data (empty data between sorted values)
def binary_search3(data, search_val):
    #print("Data: " + str(data))
    print("Search Value: " + str(search_val))
    
    first = 0
    last = len(data) - 1
    
    while first <= last:
        mid = (first + last) // 2
        if not data[mid]:
            left = mid-1
            right = mid+1
            while(True):
                if (left < first) and (right > last):
                    print ("{0} is not in the dataset".format(search_val))
                    return
                elif (right <= last) and (data[right]):
                    mid = right
                    break
                elif (left >= first) and (data[left]):
                    mid = left
                    break
                right += 1
                left -= 1
            if data[mid] == search_val:
                print("{0} found at position {1}".format(search_val, mid))
                return
            if search_val < data[mid]:
                last = mid - 1
            if search_val > data[mid]:
                first = mid + 1
    print("{0} is not in the dataset".format(search_val))
    
#DepthFirstSearch Graph Search Algorithm (5/14/2020)
#Uses a python dictionary for the graph, values as set() containing the edges
def dfs(graph, current_vertex, target_value, visited = None):
    if not visited:
        visited = []
        
    visited.append(current_vertex)
    
    if current_vertex == target_value:
        return visited
        
    for neighbor in graph[current_vertex]:
        if neighbor not in visited:
            path = dfs(graph, neighbor, target_value, visited)
            if path:
                return path
                
#BreadthFirstSearch Graph Search Algorithm (5/14/2020)
#Uses a python dictionary for the graph, values as set() containing the edges
def bfs(graph, start_vertex, target_value):
    path = [start_vertex]
    vertex_and_path = [start_vertex, path]
    bfs_queue = [vertex_and_path]
    visited = set()
    
    while bfs_queue:
        current_vertex, path = bfs_queue.pop(0)
        visited.add(current_vertex)
        
        for neighbor in graph[current_vertex]:
            if neighbor not in visited:
                if neighbor == target_value:
                    return path + [neighbor]
                else:
                    bfs_queue.append([neighbor, path + [neighbor]])
                
#Dijkstra's Algorithm (5/18/2020)
#Uses a python dictionary for the graph, with lists of tuples for the values

from heapq import heappop, heappush
from math import inf

def dijkstras(graph, start):
    distances = {}
    
    for vertex in graph:
        distances[vertex] = inf
        
    distances[start] = 0
    vertices_to_explore = [(0, start)]
    
    while vertices_to_explore:
        current_distance, current_vertex = heappop(vertices_to_explore)
        
        for neighbor, edge_weight in graph[current_vertex]:
            new_distance = current_distance + edge_weight
            
        if new_distance < distances[neighbor]:
            distances[neighbor] = new_distance
            heappush(vertices_to_explore, (new_distance, neighbor))
            
    return distances






            
    