import enum
from copy import deepcopy

#Enum for the different directions
class D(enum.IntEnum):
    up = 0
    left = 1
    down = 2
    right = 3

#Return the value in the specified direction
def move(V, i, j, dir):
    if dir == D.up and i != 0:
        return V[i-1][j]
    
    elif dir == D.left and j != 0:
        return V[i][j-1]
    
    elif dir == D.down and i != 3:
        return V[i+1][j]
    
    elif dir == D.right and j != 3:
        return V[i][j+1]
    return V[i][j]

#Return the value in the opposite direction
def move_op(V, i, j, dir):
    if dir == D.up:
        dir = D.down
    elif dir == D.left:
        dir = D.right
    elif dir == D.down:
        dir = D.up
    else:
        dir = D.left
    return move(V, i, j, dir)

#Return which action maximizes the reward
def max_action(V, i, j):
    vals = [0]*4
    for dir in D:
        vals[dir] = 0.7 * move(V, i, j, dir) + 0.2 * move_op(V, i, j, dir) + 0.1 * V[i][j]

    return vals.index(max(vals))

#Convert the numeric direction to an ascii character
def to_arrow(action):
    if action == D.up:
        return "^"
    if action == D.left:
        return "<"
    if action == D.down:
        return "v"
    return ">"

#Print V as a grid
def printV(V):
    print(" --------- --------- --------- ---------")
    for i in range(len(V)):
        print("| {0: ^7.2f} | {1: ^7.2f} | {2: ^7.2f} | {3: ^7.2f} |".format(V[i][0], V[i][1], V[i][2], V[i][3]))
        print(" --------- --------- --------- ---------")

#Print the policy as a grid
def printP(P):
    print(" --- --- --- --- ")
    for i in range(len(P)):
        print("| {} | {} | {} | {} |".format(P[i][0], P[i][1], P[i][2], P[i][3]))
        print(" --- --- --- --- ")

#Initial values
R = [ [0, 5, -2, 10], [0, 5, 0, 15], [-5, 10, 5, 0], [60, 0, 0, 5]] #Reward function
V = [ [0, 5, -2, 10], [0, 5, 0, 15], [-5, 10, 5, 0], [60, 0, 0, 5]] #Value function
P = [ [0 for i in range(4)] for j in range(4)] #Policy

#Print reward function
print("Initial reward function")
printV(V)

#Calculate V^6 and its policy
for horizon in range(6):
    newV = deepcopy(V)
    newP = deepcopy(P)

    for i in range(len(V)):
        for j in range(len(V[i])):
            dir = max_action(V, i, j)
            newP[i][j] = to_arrow(dir)
            newV[i][j] = R[i][j] + 0.7 * move(V, i, j, dir) + 0.2 * move_op(V, i, j, dir) + 0.1 * V[i][j]
    
    V = newV
    P = newP

#Print results
print("V^6")
print("Value function")
printV(V)
print("Policy")
printP(P)

#Reinitialize value function and policy
V = [ [0, 5, -2, 10], [0, 5, 0, 15], [-5, 10, 5, 0], [60, 0, 0, 5]]
P = [ [0 for i in range(4)] for j in range(4)]

#Calculate V* and its policy
discount = 0.96
epsilon = 0.0001
while True:
    newV = deepcopy(V)
    newP = deepcopy(P)
    delta = 0

    for i in range(len(V)):
        for j in range(len(V[i])):
            dir = max_action(V, i, j)
            newP[i][j] = to_arrow(dir)
            newV[i][j] = R[i][j] + discount * (0.7 * move(V, i, j, dir) + 0.2 * move_op(V, i, j, dir) + 0.1 * V[i][j])

            if abs(newV[i][j] - V[i][j]) > delta:
                delta = abs(newV[i][j] - V[i][j])
                #print(delta)
    
    V = newV
    P = newP

    if delta < epsilon * (1 - discount) / discount:
        break

#Print results
print("V*")
print("Value function")
printV(V)
print("Policy")
printP(P)