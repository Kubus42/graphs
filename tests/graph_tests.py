from unittest import TestCase
from objects.graphs import *
import numpy as np


class VertexTester(TestCase):

    def setUp(self) -> None:
        self.vertex_1 = Vertex("vertex_1")
        self.vertex_2 = Vertex("vertex_2")

    def test_add_neighbors_neither_vertex_nor_list(self):
        np.testing.assert_raises(ValueError, self.vertex_1.add_neighbors, 2)

    def test_add_neighbors_list_but_not_vertices(self):
        np.testing.assert_raises(TypeError, self.vertex_1.add_neighbors, [2])

    def test_add_neighbors_single_vertex(self):
        self.vertex_1.add_neighbors(self.vertex_2)
        np.testing.assert_array_equal(self.vertex_1.neighbors, [self.vertex_2])

    def test_add_neighbors_equal_vertex(self):
        self.vertex_1.add_neighbors([self.vertex_2, self.vertex_2])
        np.testing.assert_array_equal(self.vertex_1.neighbors, [self.vertex_2])


class GraphTester(TestCase):

    def setUp(self) -> None:
        vertices = [str(i) for i in range(4)]
        edges = {"0": [("1", 1), ("2", 1), ("3", 1)],
             "1": [("0", 1), ("3", 1)],
             "2": [("0", 1), ("3", 1)],
             "3": [("0", 1), ("1", 1), ("2", 1)]}
        self.graph = Graph(vertices=vertices, edges=edges)

    def test_number_of_edges(self):
        np.testing.assert_array_equal(self.graph.n_edges, 10)

    def test_number_of_vertices(self):
        np.testing.assert_array_equal(self.graph.n_vertices, 4)

    def test_vertex_degree(self):
        np.testing.assert_array_equal(self.graph.vertex_degree("0"), 3)
