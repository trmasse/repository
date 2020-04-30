#Data Structure Repository

#Node
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

#Linked List
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
            
#Hash Map with Separate Chaining
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
            
            
    