from collections import deque

class Graph:
    def __init__(self, vertices):
        self.num_vertices = vertices
        self.adj = [[] for _ in range(vertices)]

    def add_edge(self, src, dest):
        self.adj[src].append(dest)
        self.adj[dest].append(src)

    def view_graph(self):
        print("Graph:")
        for i in range(self.num_vertices):
            print(f"Vertex {i} -> {' '.join(map(str, self.adj[i]))}")

    def bfs(self, start_vertex):
        visited = [False] * self.num_vertices
        queue = deque()

        visited[start_vertex] = True
        queue.append(start_vertex)

        print("Breadth First Search (BFS):", end=" ")
        while queue:
            current_vertex = queue.popleft()
            print(current_vertex, end=" ")

            for neighbor in self.adj[current_vertex]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        print()

    def dfs(self, start_vertex):
        visited = [False] * self.num_vertices
        stack = []

        visited[start_vertex] = True
        stack.append(start_vertex)

        print("Depth First Search (DFS):", end=" ")
        while stack:
            current_vertex = stack.pop()
            print(current_vertex, end=" ")

            for neighbor in reversed(self.adj[current_vertex]):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        print()

# ----------- Main Logic -----------

if __name__ == "__main__":
    num_vertices = int(input("Enter the number of vertices in the graph: "))
    graph = Graph(num_vertices)

    num_edges = int(input("Enter the number of edges in the graph: "))
    print("Enter the edges (source destination):")
    for _ in range(num_edges):
        src, dest = map(int, input().split())
        graph.add_edge(src, dest)

    graph.view_graph()

    start_vertex = int(input("Enter the starting vertex for BFS and DFS: "))
    graph.bfs(start_vertex)
    graph.dfs(start_vertex)
