#Data Structure Repository
#Contents: Node, LinkedList, HashMap, Tree, MinHeap, Graph (Directed, Weighted), BubbleSort

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
        
    
            
    