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
        v1 = Vertex("vertex_1")
        v2 = Vertex("vertex_2")
        v3 = Vertex("vertex_3")
        self.graph = Graph()
        self.graph.add_vertex(v1)
        self.graph.add_vertex(v2)
        self.graph.add_vertex(v3)
        self.graph.add_edge("vertex_1", "vertex_2")
        self.graph.add_edge("vertex_1", "vertex_3")

    def test_number_of_edges(self):
        np.testing.assert_array_equal(self.graph.n_edges, 4)

    def test_number_of_vertices(self):
        np.testing.assert_array_equal(self.graph.n_vertices, 3)

    def test_vertex_degree(self):
        np.testing.assert_array_equal(self.graph.vertex_degree("vertex_1"), 2)
