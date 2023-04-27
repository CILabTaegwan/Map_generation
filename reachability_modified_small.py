import numpy as np
import math
from itertools import combinations

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def astar(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []

        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            # Child is on the closed list
            is_closed = False
            for closed_child in closed_list:
                if child == closed_child:
                    is_closed = True
            if is_closed: continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)
def find_four_direction(maze,start_p, end_p):
    four_direction = []
    for new_position in [(0,1),(1,0),(-1,0),(0,-1)]:
        tmp_list = (end_p[0]+new_position[0],end_p[1]+new_position[1])
        four_direction.append(tmp_list)

    reachability = 0
    for end_point in four_direction:

        path = astar(maze,start_p,end_point)
        if path != None:
            reachability = 1
            break
    return reachability

def get_solvability(indarray):

    solvable = True
    player1 = [(i, j) for i in range(5) for j in range(5) if indarray[i][j] == 7][0]
    player2 = [(i, j) for i in range(5) for j in range(5) if indarray[i][j] == 8][0]


    for k in range (2,6):
        #tmp_len = math.ceil(2 - k / 5) # 2,2,2,1을 뽐기 위함
        block_position = [(i, j) for i in range(5) for j in range(5) if indarray[i][j] == k]
        for m in range(len(block_position)):
            reachablilty1 = find_four_direction(indarray, player1, block_position[m])
            reachablilty2 = find_four_direction(indarray, player2, block_position[m])
            if reachablilty1*reachablilty2 != 1:
                solvable = False
                break

    return solvable


def hamming_distance (individual1,individual2):
    distance_value=0
    for j in range(5):
        for k in range(5):
            if individual1[k][j]!=individual2[k][j]:
                distance_value+=1

    return distance_value
def build_hamminglist(population):
    a = list(range (len(population)))
    b= [0 for i in range(len(population))]

    tmp_list = []
    for i in combinations(a, 2):
        value=hamming_distance(population[i[0]],population[i[1]])

        b[i[0]] += value
        b[i[1]] += value

    #sorted_b = sorted(b, reverse=True)

    sorted_index =sorted(range(len(b)),reverse=True, key=lambda k: b[k])
    '''
    for i in range(len(a)):
        for j in range(len(a)):
            if sorted_b[i] == b[j]:
                sorted_index.append(a[j])
                break
    '''

    return sorted_index
def build_hamminglist_2(population):
    a = list(range (len(population)))
    tmp_list = []
    value_list = []
    for i in combinations(a, 2):
        tmp_list.append(i)
        value=hamming_distance(population[i[0]],population[i[1]])
        value_list.append(value)

    sorted_index =sorted(range(len(value_list)),reverse=True, key=lambda k: value_list[k])

    hamming_list = tmp_list[sorted_index[0]]
    hamming_list = list(hamming_list)
    while len(hamming_list)<10:
        tmp_list2 = []
        for i in range(100):
            if i not in hamming_list:
                tmp_list2.append(i)
        new_list3 = [0 for i in range(len(tmp_list2))]
        for j in hamming_list:
            new_index = 0
            for i in tmp_list2:
                new_list3[new_index] +=hamming_distance(population[j],population[i])
                new_index+=1
        max_index = new_list3.index(max(new_list3))##인덱스 아님 i 수정 필오
        hamming_list.append(tmp_list2[max_index])

    return hamming_list
def input_or_not(population, individual):
    satisfy = 1
    for i in range(len(population)):
        value = hamming_distance(population[i],individual)
        if value ==0:
            satisfy = 0
            break
    return satisfy
def main():
    maze = [[0, 0, 0, 1, 1, 0, 0],
            [0, 8, 0, 5, 0, 0, 0],
            [0, 0, 0, 1, 0, 2, 3],
            [3, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [8, 0, 0, 1, 0, 1, 0],
            [4, 0, 4, 1, 2, 0, 7]]

    maze1 = [[5, 0, 0, 1, 0, 0,8],
            [0, 7, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 2, 3],
            [2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 4, 0],
            [4, 3, 1, 1, 2, 0, 0]]
    maze2 =[[1, 1, 1, 1, 1, 1,1],
            [1, 0, 1, 0, 0, 5, 1],
            [1, 0, 0, 1, 0, 3, 1],
            [1, 0, 0, 5, 0, 0, 1],
            [1, 1, 1, 1, 1, 4, 1]]
    maze =np.array(maze).reshape(7,7)
    maze1 = np.array(maze1).reshape(7,7)
    maze2 = np.array(maze2).reshape(7, 7)
    new_list = []
    new_list.append(maze)
    new_list.append(maze1)
    new_list.append(maze2)
    build_hamminglist(new_list)
    print(get_solvability(maze))
    #solvability = get_solvability(maze1)
    #print(input_or_not(new_list,maze2))

    #y0이 y축, 1이 x축





if __name__ == '__main__':
    main()