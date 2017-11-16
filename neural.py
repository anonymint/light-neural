"""
Neural OBject and all needed classes
"""
import numpy as np

class Node(object):
	def __init__(self, inbound_nodes=[]):
		self.inbound_nodes = inbound_nodes
		self.outbound_nodes = []
		self.value = None

		for n in self.inbound_nodes:
			n.outbound_nodes.append(self)

	def forward(self):
		"""
		forward propagation
		"""			
		raise NotImplemented

class Input(Node):
	def __init__(self):
		# only input nodes don't have inbound_nodes
		Node.__init__(self)

	def forward(self, value=None):
		if value is not None:
			self.value = value		

class Add(Node):
	def __init__(self, *inbound_nodes):
		Node.__init__(self, inbound_nodes)		

	def forward(self):
		sum_up_value = 0
		for n in self.inbound_nodes:
			sum_up_value += n.value
		self.value = sum_up_value	 			

class Multiply(Node):
	def __init__(self, *inbound_nodes):
		Node.__init__(self, inbound_nodes)		

	def forward(self):
		sum_up_value = 1
		for n in self.inbound_nodes:
			sum_up_value *= n.value
		self.value = sum_up_value	 	

class Linear(Node):
	def __init__(self, inputs, weights, bias):
		Node.__init__(self, [inputs, weights, bias])

	def forward(self):
		inputs = self.inbound_nodes[0].value
		weights = self.inbound_nodes[1].value
		bias = self.inbound_nodes[2].value				
		self.value = np.dot(inputs, weights) + bias

class Sigmoid(Node):
	def __init__(self, node):
		Node.__init__(self, [node])

	def _sigmoid(self, input):
		return 1.0 / (1.0 + np.exp(-input))		

	def forward(self):
		self.value = self._sigmoid(self.inbound_nodes[0].value)		

def topological_sort(feed_dict):
    """
    Sort the nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` Node and the value is the respective value feed to that Node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L

def forward_pass(output_node, sorted_nodes):
    """
    Performs a forward pass through a list of sorted nodes.

    Arguments:

        `output_node`: A node in the graph, should be the output node (have no outgoing edges).
        `sorted_nodes`: A topologically sorted list of nodes.

    Returns the output Node's value
    """

    for n in sorted_nodes:
        n.forward()

    return output_node.value    		