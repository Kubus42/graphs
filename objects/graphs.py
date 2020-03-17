from typing import List, Union


class Vertex:
    def __init__(self, name):
        self.name = name
        self.neighbors = []

    def add_neighbors(self, vertices: Union["Vertex", List["Vertex"]]):
        if isinstance(vertices, Vertex):
            if vertices not in self.neighbors:
                self.add_neighbors([vertices])
        elif isinstance(vertices, list):
            for v in vertices:
                if isinstance(v, Vertex):
                    if v not in self.neighbors:
                        self.neighbors.append(v)
                else:
                    raise TypeError("Neighbor must be of class 'Vertex'.")
        else:
            raise ValueError("Input needs to be a vertex or a list of vertices.")
        self.neighbors.sort()


class Graph:
    def __init__(self, vertices: list, edges: dict):
        self.vertices = vertices
        self.edges = edges
        self.n_vertices = len(self.vertices)
        self.n_edges = len([v[0] for vv in self.edges.values() for v in vv])

    def vertex_degree(self, vertex: str):
        return len([v[0] for v in self.edges[vertex]])


if __name__ == "__main__":
    v1 = Vertex("vertex_1")
    v2 = Vertex("vertex_2")
    print(v1.name)
    v1.add_neighbor(v2)
    print(v1.neighbors[0].name)


