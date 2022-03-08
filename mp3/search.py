# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

from collections import deque
import heapq

"""
This is the main entry point for MP3. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)


# Feel free to use the code below as you wish
# Initialize it with a list/tuple of objectives
# Call compute_mst_weight to get the weight of the MST with those objectives
# TODO: hint, you probably want to cache the MST value for sets of objectives you've already computed...
# Note that if you want to test one of your search methods, please make sure to return a blank list
#  for the other search methods otherwise the grader will not crash.
class MST:
    def __init__(self, objectives):
        self.elements = {key: None for key in objectives}

        # TODO: implement some distance between two objectives
        # ... either compute the shortest path between them, or just use the manhattan distance between the objectives
        self.distances   = {
                (i, j): self.distance_between_objectives(i,j)
                for i, j in self.cross(objectives)
                    
            }

    # Prim's algorithm adds edges to the MST in sorted order as long as they don't create a cycle
    def compute_mst_weight(self):
        weight      = 0
        for distance, i, j in sorted((self.distances[(i, j)], i, j) for (i, j) in self.distances):
            if self.unify(i, j):
                weight += distance
        return weight

    # helper checks the root of a node, in the process flatten the path to the root
    def resolve(self, key):
        path = []
        root = key
        while self.elements[root] is not None:
            path.append(root)
            root = self.elements[root]
        for key in path:
            self.elements[key] = root
        return root

    # helper checks if the two elements have the same root they are part of the same tree
    # otherwise set the root of one to the other, connecting the trees
    def unify(self, a, b):
        ra = self.resolve(a)
        rb = self.resolve(b)
        if ra == rb:
            return False
        else:
            self.elements[rb] = ra
            return True

# helper that gets all pairs i,j for a list of keys
    def cross(self, keys):
        return (x for y in (((i, j) for j in keys if i < j) for i in keys) for x in y)

    def distance_between_objectives(self, i,j):
        k = 0
        distance = abs(i[0] - j[0]) + abs(i[1] - j[1])
        #print("i: ", i, " j: " , j)
        #print("distance: ", distance)
        return distance


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    current = maze.start
    dictParent = {}
    queue = deque([])
    explored = []
    path = []
    queue.append(maze.start)
    dictParent[maze.start] = "Start bro"
    obj = maze.waypoints[0]


    while queue:
        current = queue.popleft()
        if current == obj:
            while dictParent[current] != "Start bro":
                path.append(current)
                current = dictParent[current]
            path.append(current)
            path.reverse()
            return path


        explored.append(current)
        for neighbor in maze.neighbors(current[0], current[1]):
            if neighbor not in explored and neighbor not in queue:
                queue.append(neighbor)
                dictParent[neighbor] = current
    return path
def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    path = []
    explored = []
    dictParent = {}
    queue = []
    heuristic = {}

    heuristic[maze.start] = 0
    heapq.heappush(queue, (0,maze.start))

    while queue:
        current = heapq.heappop(queue)
        if current[1] == maze.waypoints[0]:
            break
        explored.append(current[1])
        for neighbor in maze.neighbors(current[1][0], current[1][1]):
            if neighbor not in explored and neighbor not in queue:
                manDist = abs(neighbor[0] - maze.waypoints[0][0]) + abs(neighbor[1] - maze.waypoints[0][1])
                heuristic[neighbor] = heuristic[current[1]] + 1
                dist = manDist + heuristic[neighbor]
                explored.append(neighbor)
                heapq.heappush(queue, (dist, neighbor))
                dictParent[neighbor] = current

    while current[0] != 0:
        path.append(current[1])
        current = dictParent[current[1]]      
    path.append(current[1])
    path.reverse()
    return path

def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    path = []
    explored = {}
    dictParent = {}
    queue = []
    f_n = {}
    mst ={}

    waypoints_left = maze.waypoints
    initialDist = 0
    start = (waypoints_left, maze.start, initialDist)
    startExp = (waypoints_left, maze.start)
    explored[startExp] = 0
    f_n[start] = 0
    heapq.heappush(queue, start)
    
    while queue:
        curr = heapq.heappop(queue)
        #print("CURR: ", curr)

        temp1 = (curr[0], curr[1])
        temp2 = f_n[curr]

        if curr[1] in curr[0]:
            objectives_list = list(curr[0])
            objectives_list.remove(curr[1])
            tupleOBJ = tuple(objectives_list)
            curr = (tupleOBJ, curr[1], curr[2])
            state = (curr[0], curr[1])
            explored[state] = temp1
            f_n[curr] = temp2

        #how many waypts left if 0 return path
        if len(curr[0]) <= 0:
            path.append(curr[1])
            state2 = (curr[0], curr[1])
            while explored[state2] != 0:
                state2 = explored[state2]
                if state2[1] != path[-1]:
                    path.append(state2[1])


            path.reverse()
            print(path)
            print("length of PATH: ", len(path))
            return path

        for neighbor in maze.neighbors(curr[1][0], curr[1][1]):
            minDist = abs(neighbor[0] - curr[0][0][0]) + abs(neighbor[1] - curr[0][0][1])
            for waypt in curr[0]:
                dist = abs(neighbor[0] - waypt[0]) + abs(neighbor[1] - waypt[1])
                if (dist < minDist):
                    minDist = dist

            #create mst for remaining waypts
            mstObj = mst.get(curr[0], 0)
            if mstObj == 0:
                mstObj = MST(curr[0])
            mstLen = mstObj.compute_mst_weight()

            heuristic = minDist + mstLen

            #calculate f(n) for each node
            dist = f_n[curr] + heuristic + 1
            neighState = (curr[0], neighbor, dist)
            state3 = (neighState[0], neighState[1])
            f_n[neighState] = f_n[curr] + 1

            if state3 not in explored:
                heapq.heappush(queue, neighState)
                explored[state3] = (curr[0], curr[1])

   

    return path

def fast(maze):
    """
    Runs suboptimal search algorithm for extra credit/part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    path = []
    explored = {}
    dictParent = {}
    queue = []
    f_n = {}
    mst ={}

    waypoints_left = maze.waypoints
    initialDist = 0
    start = (waypoints_left, maze.start, initialDist)
    startExp = (waypoints_left, maze.start)
    explored[startExp] = 0
    f_n[start] = 0
    heapq.heappush(queue, start)
    
    while queue:
        curr = heapq.heappop(queue)
        #print("CURR: ", curr)

        temp1 = (curr[0], curr[1])
        temp2 = f_n[curr]

        if curr[1] in curr[0]:
            objectives_list = list(curr[0])
            objectives_list.remove(curr[1])
            tupleOBJ = tuple(objectives_list)
            curr = (tupleOBJ, curr[1], curr[2])
            state = (curr[0], curr[1])
            explored[state] = temp1
            f_n[curr] = temp2

        #how many waypts left if 0 return path
        if len(curr[0]) <= 0:
            path.append(curr[1])
            state2 = (curr[0], curr[1])
            while explored[state2] != 0:
                state2 = explored[state2]
                if state2[1] != path[-1]:
                    path.append(state2[1])


            path.reverse()
            print(path)
            print("length of PATH: ", len(path))
            return path

        for neighbor in maze.neighbors(curr[1][0], curr[1][1]):
            minDist = abs(neighbor[0] - curr[0][0][0]) + abs(neighbor[1] - curr[0][0][1])
            for waypt in curr[0]:
                dist = abs(neighbor[0] - waypt[0]) + abs(neighbor[1] - waypt[1])
                if (dist < minDist):
                    minDist = dist

            #create mst for remaining waypts
            mstObj = mst.get(curr[0], 0)
            if mstObj == 0:
                mstObj = MST(curr[0])
            mstLen = mstObj.compute_mst_weight()

            heuristic = minDist + mstLen
            heuristic = heuristic * 1.5

            #calculate f(n) for each node
            dist = f_n[curr] + heuristic + 1
            neighState = (curr[0], neighbor, dist)
            state3 = (neighState[0], neighState[1])
            f_n[neighState] = f_n[curr] + 1

            if state3 not in explored:
                heapq.heappush(queue, neighState)
                explored[state3] = (curr[0], curr[1])

   

    return path


