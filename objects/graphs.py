from typing import List, Union, Dict


class Vertex:
    def __init__(self, name: str):
        self.name = name
        self.neighbors = []

    def __lt__(self, vertex: "Vertex"):
        return self.name < vertex.name

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
    def __init__(self):
        self.vertices: Dict[str, Vertex] = {}
        self.n_vertices = 0
        self.n_edges = 0

    def add_vertex(self, vertex: Vertex):
        if isinstance(vertex, Vertex):
            if vertex.name not in self.vertices:
                self.vertices[vertex.name] = vertex
                self.n_vertices += 1
            else:
                print("There already exists a vertex with this name.")
        else:
            raise ValueError("Vertices must be of class 'Vertex'.")

    def add_edge(self, u: str, v: str):
        if u in self.vertices and v in self.vertices:
            if u not in self.vertices[v].neighbors:
                self.vertices[v].add_neighbors(self.vertices[u])
                self.n_edges += 1
            if v not in self.vertices[u].neighbors:
                self.vertices[u].add_neighbors(self.vertices[v])
                self.n_edges += 1
        else:
            raise ValueError("Either 'u' or 'v' is not a vertex of the graph.")

    def vertex_degree(self, vertex: str):
        return len(self.vertices[vertex].neighbors)

    def print_graph(self):
        for vertex in self.vertices.values():
            print("Vertex: " + vertex.name + ", neighbors: " + str([v.name for v in vertex.neighbors]))


if __name__ == "__main__":
    v1 = Vertex("vertex_1")
    v2 = Vertex("vertex_2")
    v3 = Vertex("vertex_3")

    graph = Graph()
    graph.add_vertex(v1)
    graph.add_vertex(v2)
    graph.add_vertex(v3)

    graph.add_edge("vertex_1", "vertex_2")
    graph.add_edge("vertex_1", "vertex_3")

    graph.print_graph()


