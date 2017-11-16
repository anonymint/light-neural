from neural import *

x, y, z = Input(), Input(), Input()

f = Add(x, y, z)
f_mul = Multiply(x, y, z)

feed_dict = {x: 4, y: 5, z: 10}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)

print("{} + {} + {} ==> ".format(feed_dict[x], feed_dict[y], feed_dict[z]), output)

output = forward_pass(f_mul, graph)
print("{} * {} * {} ==> ".format(feed_dict[x], feed_dict[y], feed_dict[z]), output)

# Let make it Linear with Sigmoid
X, W, b = Input(), Input(), Input()

f = Linear(X, W, b)
g = Sigmoid(f)

X_ = np.array([[-1.5, 2.4], [1., 1.1]])
W_ = np.array([[3.3, -3.3], [2.2, -2.2]])
b_ = np.array([-3., -5])

feed_dict = {X: X_, W: W_, b: b_}

graph = topological_sort(feed_dict)
output = forward_pass(g, graph)

print(output)