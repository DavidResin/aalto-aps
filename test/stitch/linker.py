import cv2
import numpy as np
from tree import ImageTree

def solve(array, index=0):
	shape = array.shape
	edge_array = np.full(shape, False)
	cost_array = np.empty(shape)

	for (x, y), elem in np.ndenumerate(array):
		if elem is None or elem[index] is None:
			cost_array[x][y] = 0
		else:
			cost_array[x][y] = len(elem[index])
	
	sets = []

	for i in range(len(array)):
		sets.append({i})

	while len(sets[0]) != len(sets) and np.sum(cost_array) > 0:
		pos = find_biggest(edge_array, cost_array)
		(x, y) = pos

		if sets[x] != sets[y]:
			edge(pos, edge_array, cost_array)
			temp = sets[x] | sets[y]

			for e in temp:
				sets[e] = temp
		else:
			update(cost_array, pos, 0)

	return edge_array

def update(array, pos, value):
	(x, y) = pos
	array[x][y] = value
	array[y][x] = value

def edge(pos, edge_array, cost_array):
	update(cost_array, pos, 0)
	update(edge_array, pos, True)

def find_biggest(edge_array, cost_array):
	pos = np.unravel_index(cost_array.argmax(), cost_array.shape)
	return pos

def edgesToTree(firstelem, array, edge_table, data_table):
	tree = ImageTree(firstelem)
	edge_rec(tree, array, edge_table, data_table)
	return tree

def edge_rec(tree, array, edge_table, data_table):
	index = tree.data.index
	line = edge_table[index].tolist()

	while any(line):
		curr_idx = line.index(True)
		pos = (index, curr_idx)
		update(edge_table, pos, False)
		_, matrix, _ = data_table[index, curr_idx]
		image = array[curr_idx]
		image.matrix = np.dot(matrix, tree.data.matrix)
		newTree = tree.makeChildOf(image)
		line = edge_table[index].tolist()
		edge_rec(newTree, array, edge_table, data_table)