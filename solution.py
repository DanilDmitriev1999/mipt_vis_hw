import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

class LayeredTree():
    def __init__(self, filename):
        self.filename = filename
        self.coordinates = {}
        self.tree = nx.read_graphml(self.filename)

    def calculate_coordinates(self, tree, node, previous_node):
        offset = 0
        maxx = 0
        sum = 0
        count = 0
        child_y = 0

        if len(list(nx.neighbors(self.tree, node))) == 0:
            self.tree.nodes[node]['corr_x'] = 0
            self.tree.nodes[node]['corr_y'] = child_y
            return

        for current_node in list(nx.neighbors(self.tree, node)):
            if current_node != previous_node:
                self.calculate_coordinates(self.tree, current_node, node)

        for current_node in nx.neighbors(self.tree, node):
            if current_node != previous_node:
                maxx = self.move_tree(self.tree, current_node, node, offset, maxx)
                offset = maxx + offset + 2
                child_y = max(self.tree.nodes[current_node]['corr_y'], child_y)
                count += 1
                sum += self.tree.nodes[current_node]['corr_x']
        tree.nodes[node]['corr_x'] = int(sum / count)
        tree.nodes[node]['corr_y'] = child_y + 1
        x = 0

    def move_tree(self, tree, node, previous_node, offset, maxx):
        self.tree.nodes[node]['corr_x'] += offset
        maxx = max(tree.nodes[node]['corr_x'], maxx)
        if len(list(nx.neighbors(tree, node))) == 0:
            return maxx
        for current_node in list(nx.neighbors(self.tree, node)):
            if current_node != previous_node:
                maxx = max(maxx, self.move_tree(self.tree, current_node, node, offset, maxx))
        return maxx

    def draw(self):
        self.calculate_coordinates(self.tree, 'n0', 'n0')

        coordinates = {}
        for node in list(self.tree.nodes):
            coordinates[node] = (self.tree.nodes[node]['corr_x'], self.tree.nodes[node]['corr_y'])

        plt.figure(figsize=(20, 10))

        nx.draw(self.tree, pos=coordinates, with_labels=True)
        plt.show()


if __name__ == '__main__':
    tree = LayeredTree('../dt/tree-42n.xml')
    tree.draw()