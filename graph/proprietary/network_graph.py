class Graph:
    def __init__(self):
        self.graph_dict = {}
        self.edges = set([])

    def add_node(self, node):
        if node not in self.graph_dict:
            self.graph_dict[node] = set([])

    def get_nodes(self):
        return self.graph_dict.keys()

    def add_edge(self, node, neighbour):
        self.edges.add((node, neighbour))
        if node not in self.graph_dict:
            self.graph_dict[node] = set([neighbour])
        else:
            self.graph_dict[node].add(neighbour)

    def has_edge(self, node, neighbour):
        return (node, neighbour) in self.edges

    def show_edges(self):
        for node in self.graph_dict:
            for neighbour in self.graph_dict[node]:
                print("(", node, ", ", neighbour, ")")

    def find_path(self, start, end, path=[]):
        path = path + [start]
        if start == end:
            return path
        for node in self.graph_dict[start]:
            if node not in path:
                newPath = self.find_path(node, end, path)
                if newPath:
                    return newPath
                return None

    def bfs(self, s):
        result = []
        visited = {}
        for i in self.graph_dict:
            visited[i] = False
        queue = [s]
        visited[s] = True
        while len(queue) != 0:
            s = queue.pop(0)
            for node in self.graph_dict[s]:
                if not visited[node]:
                    visited[node] = True
                    queue.append(node)
            # print(s, end=" ")
            result.append(s)

        return result

    def all_paths(self, start, end, path=[]):
        path = path + [start]
        if start == end:
            return [path]
        paths = []
        for node in self.graph_dict[start]:
            if node not in path:
                newpaths = self.all_paths(node, end, path)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths

    def shortest_path(self, start, end, path=[]):
        path = path + [start]
        if start == end:
            return path
        shortest = None
        for node in self.graph_dict[start]:
            if node not in path:
                newpath = self.shortest_path(node, end, path)
                if newpath:
                    if not shortest or len(shortest) > len(newpath):
                        shortest = newpath
        return shortest

    def dfs(self, s):
        result = []
        visited = {}
        for i in self.graph_dict:
            visited[i] = False
        stack = [s]
        visited[s] = True
        while stack:
            n = stack.pop(len(stack) - 1)
            for i in self.graph_dict[n]:
                if not visited[i]:
                    stack.append(i)
                    visited[i] = True
            #print(n)

            result.append(n)

        return result
