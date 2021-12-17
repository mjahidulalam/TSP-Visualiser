import math
import numpy as np
import plotly.graph_objects as go
import time
from numba import njit

class Solver:
    def __init__(self):
        pass
    
    def solve_it(self, node_input, method_input, local_search, animate_input, **style):
        self.points = np.random.uniform(low=5, high=105, size=(int(node_input),2))

        distances = np.zeros((len(self.points), len(self.points)))
        for i in range(len(self.points)):
            for j in range(i+1, len(self.points)):
                distances[i,j] = self.length(self.points[i], self.points[j])

        self.distances = distances + distances.T

        self.animate = animate_input
        self.method = method_input
        self.style = style
        
        if method_input == "NN":
            solution = self.nearest_neighbor(self.points, self.distances)
            data = self.frames(solution)
        elif method_input == "NI":
            solution, output = self.nearest_insertion(self.points, self.distances)
            data = self.frames(output)
        elif method_input == "FI":
            solution, output = self.farthest_insertion(self.points, self.distances)
            data = self.frames(output)
        else:
            raise ValueError("Incorrect method")

        if local_search == "2OPT" and len(solution) <= 500:
            improvement_threshold = 0.001*len(solution)
            solution = self.two_opt(np.array(solution), self.points, improvement_threshold)
            data = self.frames(solution, local_search=True, data=data)

        return self.points, solution, data

    def length(self, point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    @staticmethod
    @njit
    def nearest_neighbor(points, distances):
        p = np.arange(len(points))
        solution = np.zeros_like(p)
        
        i = 0
        for ii in range(len(points)):
            dist = np.array([distances[i,j] for j in p])
            index = np.argmin(dist)
            solution[ii] = p[index]
            i = p[index]
            p = np.delete(p, index)

        return solution

    @staticmethod
    @njit
    def nearest_insertion(points, distances):
        solution = [0,0]
        output = []
        p = [i for i in range(1,len(points))]

        while len(p) != 0:
            dist = []
            for i in solution[:-1]:
                dists = [distances[i][j] for j in p]
                dist.append(p[dists.index(min(dists))])

            objs = []
            for i, d in enumerate(dist):
                a = solution[i]
                b = solution[i+1]
                c = solution[i-1]
                val = distances[a, d] + distances[b, d] - distances[a, b]
                val2 =distances[a, d] + distances[c, d] - distances[a, c]
                if val < val2:
                    objs.append((1, val))
                else:
                    objs.append((0, val2))

            mini = sorted(objs,key=lambda x: x[1], reverse=False)[0]
            index = objs.index(mini)
            solution.insert(max(1,index+mini[0]), dist[index])
            output.append(list(solution))

            p.remove(dist[index])

        return solution[:-1], output

    @staticmethod
    # @njit
    def farthest_insertion(points, distances):
        solution = [0,0]
        output = []
        p = [i for i in range(1,len(points))]

        while len(p) != 0:
            dist2 = []
            # dist = []
            for n, i in enumerate(solution[:-1]):
                dists = [distances[i][j] for j in p]
                dist2.append(max(dists))

            d = dist2.index(max(dist2))
            dist2 = None
            dist = []
            
            for i in solution[:-1]:
                dists = [distances[i][j] for j in p]
                dist.append(p[dists.index(max(dists))])
                # dist.append(p[np.argmax(dists)])
                # print(dists)



            total = np.array([distances[s, solution[min(i+1,len(solution)-1)]] for i, s in enumerate(solution)]).sum()

            # d = dist[dist2.index(max(dist2))]
            # d = dist[np.argmax(dist2)]
            d = dist[d]
            objs=[]
            for i, a in enumerate(solution[:-1]):
                b = solution[i+1]
                c = solution[i-1]
                val = distances[a, d] + distances[b, d] - distances[a, b] + total
                val2 = distances[a, d] + distances[c, d] - distances[a, c] + total

                if val < val2:
                    objs.append((i+1, d, val))
                else:
                    objs.append((i, d, val2))

            mini = sorted(objs,key=lambda x: x[2], reverse=False)[0]
            solution.insert(max(1,mini[0]), mini[1])
            output.append(list(solution))
            p.remove(d)
            
        return solution[:-1], output

    @staticmethod
    @njit
    def two_opt(route, points, improvement_threshold = 0.001):
        improvement_factor = 1
        best_distance = np.array([np.linalg.norm(points[route[p]]-points[route[p-1]]) for p in range(len(route))]).sum()

        while improvement_factor > improvement_threshold:
            initial_distance = best_distance
            for node1 in range(1,len(route)-2):
                for node2 in range(node1+1,len(route)):
                    new_route = np.concatenate((route[0:node1],route[node2:-len(route)+node1-1:-1],route[node2+1:len(route)]))
                    new_distance = np.array([np.linalg.norm(points[new_route[p]]-points[new_route[p-1]]) for p in range(len(new_route))]).sum()
                    if new_distance < best_distance:
                        route = new_route
                        best_distance = new_distance
            improvement_factor = 1 - best_distance/initial_distance
        
        return route

    def frames(self, solution, local_search=False, data = None):

        if local_search == True:
            plot = np.array([[self.points[i][0], self.points[i][1]] for i in solution] + [[self.points[0][0], self.points[0][1]]])
            data += [data[-1] for _ in range(10)]
            data.append(go.Frame(data=[go.Scatter(x=[self.points[0,0]], y=[self.points[0,1]], mode='markers', marker=dict(color=self.style['HOME_MARKER_COLOUR'], size=self.style['HOME_MARKER_SIZE'])),
                    go.Scatter(x=self.points[1:,0], y=self.points[1:,1], mode='markers', marker=dict(color=self.style['NODE_MARKER_COLOUR'], size=self.style['NODE_MARKER_SIZE'])), 
                    go.Scatter(x=plot[:,0], y=plot[:,1], mode="lines", line=dict(color=self.style['LINE_COLOUR']))]))
            return data

        data = []
        if len(self.animate) != 0:
            if self.method == "NN":
                plot = np.array([[self.points[i][0], self.points[i][1]] for i in solution] + [[self.points[0][0], self.points[0][1]]])
                for i in range(len(plot)+1):
                    x = plot[:i,0]
                    y = plot[:i,1]
                    curr_frame = go.Frame(data=[go.Scatter(x=[self.points[0,0]], y=[self.points[0,1]], mode='markers', marker=dict(color=self.style['HOME_MARKER_COLOUR'], size=self.style['HOME_MARKER_SIZE'])),
                                go.Scatter(x=self.points[1:,0], y=self.points[1:,1], mode='markers', marker=dict(color=self.style['NODE_MARKER_COLOUR'], size=self.style['NODE_MARKER_SIZE'])), 
                                go.Scatter(x=x, y=y, mode="lines", line=dict(color=self.style['LINE_COLOUR']))])
                    data.append(curr_frame)
            elif self.method == "NI" or self.method == "FI":
                for s in solution:
                    plot = np.array([[self.points[i][0], self.points[i][1]] for i in s])
                    data.append(go.Frame(data=[go.Scatter(x=[self.points[0,0]], y=[self.points[0,1]], mode='markers', marker=dict(color=self.style['HOME_MARKER_COLOUR'], size=self.style['HOME_MARKER_SIZE'])),
                        go.Scatter(x=self.points[1:,0], y=self.points[1:,1], mode='markers', marker=dict(color=self.style['NODE_MARKER_COLOUR'], size=self.style['NODE_MARKER_SIZE'])), 
                        go.Scatter(x=plot[:,0], y=plot[:,1], mode="lines", line=dict(color=self.style['LINE_COLOUR']))]))
        else:
            plot = np.array([[self.points[i][0], self.points[i][1]] for i in solution] + [[self.points[0][0], self.points[0][1]]], dtype=object)
            data = [go.Frame(data=[go.Scatter(x=[self.points[0,0]], y=[self.points[0,1]], mode='markers', marker=dict(color=self.style['HOME_MARKER_COLOUR'], size=self.style['HOME_MARKER_SIZE'])),
                    go.Scatter(x=self.points[1:,0], y=self.points[1:,1], mode='markers', marker=dict(color=self.style['NODE_MARKER_COLOUR'], size=self.style['NODE_MARKER_SIZE'])), 
                    go.Scatter(x=plot[:,0], y=plot[:,1], mode="lines", line=dict(color=self.style['LINE_COLOUR']))])]

        
        return data