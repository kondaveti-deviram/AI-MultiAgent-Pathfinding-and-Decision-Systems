import random
import time
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
import copy

# Function to print the ship layout
def print_ship(ship, D):
    for row in ship:
        print(" ".join(row))
    print()

# Function to visualise the ship layout and game status
def visualize_ship(ship, D, goal, status, bot_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # Create a figure with two subplots

    # Creating a matrix for visualization
    visual_matrix = np.zeros((D, D, 3), dtype=np.uint8)

    # Mapping ship elements to colours
    for i in range(D):
        for j in range(D):
            if ship[i][j] == '0':
                visual_matrix[i, j] = [0, 0, 0]  # Black for 'X'
            elif ship[i][j] in ('1','C'):
                visual_matrix[i, j] = [255, 255, 255]  # White for '0'
            elif ship[i][j] == 'A':
                visual_matrix[i, j] = [255, 0, 0]  # Red for Alien - A
            elif ship[i][j] == 'B':
                visual_matrix[i, j] = [102, 178, 255]  # Blue for Bot - B

    # Highlighting the captain in green color
    ax1.plot(goal[1], goal[0], marker='o', markersize=10, color='green')

    # Highlighting the bot's path in blue color
    if bot_path:
        bot_path = np.array(bot_path)
        ax1.plot(bot_path[:, 1], bot_path[:, 0], color='blue', linewidth=2)

    ax1.imshow(visual_matrix, interpolation='nearest')
    ax1.set_title('Matrix Visualization')

    # Displaying game status in the second subplot
    if status == "SUCCESS":
        ax2.text(0.5, 0.5, 'BOT SAVED THE CAPTAIN!', horizontalalignment='center', verticalalignment='center',
                 fontsize=20, color='green')
    elif status == "FAILURE":
        ax2.text(0.5, 0.5, 'GAME OVER!\n THE ALIEN ATTACKED THE BOT', horizontalalignment='center',
                 verticalalignment='center', fontsize=20, color='red')
    elif status == "NO_PATH":
        ax2.text(0.5, 0.5, 'NO PATH FOUND FROM \n BOT TO CAPTAIN', horizontalalignment='center',
                 verticalalignment='center', fontsize=20, color='red')

    ax2.set_axis_off()
    plt.show(block=False)

# Function to check open neighbors around a cell
def check_open_neighbours(ship, D, row, col):
    result = []
    if row - 1 >= 0 and ship[row - 1][col] in ('1','C','B'):
        result.append((row-1,col))
    if row + 1 < D and ship[row + 1][col] in ('1','C','B'):
        result.append((row+1,col))
    if col - 1 >= 0 and ship[row][col - 1] in ('1','C','B'):
        result.append((row,col-1))
    if col + 1 < D and ship[row][col + 1] in ('1','C','B'):
        result.append((row,col+1))
    return result

# Function to check blocked neighbors around a cell
def check_blocked_neighbours(ship, D, row, col):
    result = []
    if row - 1 >= 0 and ship[row - 1][col] == "0":
        result.append((row-1,col))
    if row + 1 < D and ship[row + 1][col] == "0":
        result.append((row+1,col))
    if col - 1 >= 0 and ship[row][col - 1] == "0":
        result.append((row,col-1))
    if col + 1 < D and ship[row][col + 1] == "0":
        result.append((row,col+1))
    return result

# Function to get neighbours around the aliens
def get_alien_neighbours(ship, D, aliens_pos):
    result = []
    for alien in aliens_pos:
            x,y = alien
            if(x-1 >= 0 ):
                result.append((x-1,y))
            if(x+1 < D ):
                result.append((x+1,y))
            if(y-1 >= 0 ):
                result.append((x,y-1))
            if(y+1 < D ):
                result.append((x,y+1))
    return result

# Function to get blocked cells with only one open neighbor
def get_blocked_cells_with_one_open_neighbor(ship, D):
    result = []
    for i in range(D):
        for j in range(D):
            if ship[i][j] == '0':
                neighbours = check_open_neighbours(ship, D, i, j)
                if len(neighbours) == 1:
                    result.append((i, j))
    return result


# Function to get open cells in the ship
def get_open_cells(ship, D):
    result = []
    for i in range(D):
        for j in range(D):
            if ship[i][j] == '1':
                result.append((i, j))
    return result

# Function to get dead ends in the ship
def get_dead_ends(ship, D):
    result = []
    for i in range(D):
        for j in range(D):
            if ship[i][j] == '1':
                neighbours = check_open_neighbours(ship, D, i, j)
                if len(neighbours) == 1:
                    result.append((i, j))
    return result

# Function to generate a random ship layout
def generate_ship_layout(D):
    ship = [['0' for _ in range(D)] for _ in range(D)]
    random.seed()

    # Randomly unlocking a cell on the ship
    start_row = random.randint(0, D - 1)
    start_col = random.randint(0, D - 1)
    ship[start_row][start_col] = '1'

    while True:

        blocked_cells = get_blocked_cells_with_one_open_neighbor(ship, D)
        if not blocked_cells:
            break
        index = random.randint(0, len(blocked_cells) - 1)
        new_x, new_y = blocked_cells[index]
        ship[new_x][new_y] = '1'

    dead_ends = get_dead_ends(ship, D)
    random.seed()

    # Opening the closed neighbors of approximately half of the dead-end cells at random.
    for _ in range(len(dead_ends) // 2):
        index = random.randint(0, len(dead_ends) - 1)
        new_x, new_y = dead_ends[index]
        if ship[new_x][new_y] == '1':
            blockedNeighbours = check_blocked_neighbours(ship, D, new_x, new_y)
            if len(blockedNeighbours) >= 1:
                index = random.randint(0, len(blockedNeighbours) - 1)
                new_x, new_y = blockedNeighbours[index]
                ship[new_x][new_y] = '1'

    return ship

"""# Heuristic and A-Star Algorithm"""

# Heuristic function is the Manhattan distance between the two points
def heuristic(a,b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# A-star algorithm to find the path from start to goal
def get_bot_path_a_star(ship, D, start, goal):
    fringe = PriorityQueue()
    fringe.put((0,start))
    dist = { start:0 }
    prev = {}
    while not fringe.empty():
        _, curr = fringe.get()
        if curr == goal :
            path = []
            while curr in prev:
                path.append(curr)
                curr = prev[curr]
            path.append(start)
            return path[::-1]

        x,y = curr
        neighbors = check_open_neighbours(ship, D, x, y)

        for neighbor in neighbors:
            tempDist = dist[curr] + 1
            if neighbor not in dist or tempDist < dist[neighbor] :
                dist[neighbor] = tempDist
                prev[neighbor] = curr
                priority = dist[neighbor] +heuristic(neighbor,goal)
                fringe.put((priority,neighbor))
    return None

"""# Moving Aliens and Checking Game Status"""

# Function to move all aliens 1 step either up/down/right/left
def move_aliens(ship,D, aliens_pos):
    new_alien_pos = []
    for alien in aliens_pos:
        x,y = alien
        next_move=alien
        neighbors = check_open_neighbours(ship, D, x, y)

        # Alien moves only if it has any open cells as neighbors
        next_move  = random.choice(neighbors) if neighbors else (x, y)
        new_x,new_y = next_move
        ship[new_x][new_y] = 'A'
        ship[x][y] = '1'
        new_alien_pos.append(next_move)
    return new_alien_pos

# Function to check the game status after the bot moves
def check_after_bot_moves(ship,D,goal,bot_pos,aliens_pos):

    # If bot reached goal and alien is not present at goal
    if(bot_pos == goal and bot_pos not in aliens_pos):
        print("Success!!!")
        return "SUCCESS",None

    # If bot reached a cell where alien is present
    elif(bot_pos in aliens_pos):
        print("GAME OVER!!!!")
        return "FAILURE",bot_pos

    # Move the aliens to a random neighboring cell
    aliens_pos = move_aliens(ship,D,aliens_pos)

    # If aliens reach the bot
    if bot_pos in aliens_pos:
        print("GAME OVER!!!!")
        return "FAILURE",bot_pos

    return "NEXT",aliens_pos

"""# BOT 1"""

# Function of Bot1 stimulation
def simulate_bot1(ship,D,start,goal,aliens_pos):
    D = len(ship)
    bot_path = get_bot_path_a_star(ship, D, start , goal)
    final_bot_path =[start]
    i=0
    while True:

        # Visualize the ship grid
        plt.close()
        visualize_ship(ship,D,goal,"",final_bot_path)
        plt.pause(1)

        # plt.pause(0.5)
        if(bot_path):

            # Move the bot if there is a path to the captain
            if i<len(bot_path)-1:
                b_x,b_y = bot_path[i]
                new_x,new_y = bot_path[i+1]
                ship[b_x][b_y] = '1'
                ship[new_x][new_y] = 'B'
                final_bot_path.append((new_x,new_y))

            # Check if bot gets attacked by aliens and then move the aliens
            status,aliens_pos = check_after_bot_moves(ship,D,goal,(new_x,new_y),aliens_pos)
            i+=1


            if status == "NEXT":
                continue
            elif status in ("SUCCESS","FAILURE"):
                visualize_ship(ship,D,goal,status,final_bot_path)
                plt.pause(5)
                break
        else :

            # if no path is found by A*
            status = "NO_PATH"
            visualize_ship(ship,D,goal,status,bot_path)
            print("No  path found!!")
            plt.pause(5)
            break

    return status,i

"""# BOT 2"""

# Function of Bot2 stimulation
def simulate_bot2(ship,D,start,goal,aliens_pos):
    print("Start: ",start," Goal: ",goal)
    i=0
    count = 0
    bot_path = None
    final_bot_path = [start]
    # Constraining the loop to a maximum of 1000 steps
    while i < 1000:

        plt.close()
        # Visualize the ship grid
        visualize_ship(ship,D,goal,"",final_bot_path)
        plt.pause(1)

        # Finding the path from the bot to the captain for the current ship configuration at each iteration
        bot_path = get_bot_path_a_star(ship, D, start , goal)

        b_x,b_y = start
        new_x,new_y = start
        if(bot_path):

            # Move the bot
            new_x,new_y = bot_path[1]
            ship[b_x][b_y] = '1'
            ship[new_x][new_y] = 'B'
            final_bot_path.append((new_x,new_y))

        # Check if bot gets attacked by aliens and then move the aliens
        status,aliens_pos = check_after_bot_moves(ship,D,goal,(new_x,new_y),aliens_pos)
        i+=1
        if status == "NEXT":
            start = (new_x,new_y)
            continue
        elif status in ("SUCCESS","FAILURE"):
            visualize_ship(ship,D,goal,status,final_bot_path)
            plt.pause(10)
            return status,i


    return "FAILURE" , i

"""# BOT 3"""

# Function of Bot3 stimulation
def simulate_bot3(ship,D,start,goal,aliens_pos):
    print("Start: ",start," Goal: ",goal)
    i=0
    bot_path = None
    final_bot_path=[start]
    # Constraining the loop to a maximum of 1000 steps
    while i < 1000:

        # Visualize the ship grid
        visualize_ship(ship,D,goal,"",final_bot_path)
        plt.pause(1)

        # Add the surrounding cells of aliens also as blocked cells
        buffer = get_alien_neighbours(ship, D, aliens_pos)
        new_ship = [row[:] for row in ship]
        for cell in buffer:
            x,y = cell
            new_ship[x][y] = '0'

        # Finding the path from the bot to the captain for the current ship configuration at each iteration
        bot_path = get_bot_path_a_star(new_ship, D, start , goal)

        b_x,b_y = start
        new_x,new_y = start
        if(bot_path):

            # Move the bot
            new_x,new_y = bot_path[1]
            ship[b_x][b_y] = '1'
            ship[new_x][new_y] = 'B'
            final_bot_path.append((new_x,new_y))

        # Check if bot gets attacked by aliens and then move the aliens
        status,aliens_pos = check_after_bot_moves(ship,D,goal,(new_x,new_y),aliens_pos)
        i+=1
        if status == "NEXT":
            start = (new_x,new_y)

            continue
        elif status in ("SUCCESS","FAILURE"):
            visualize_ship(ship,D,goal,status,final_bot_path)
            plt.pause(5)

            return status, i


    return "FAILURE", i

"""# BOT 4"""

def heuristic_avoiding_aliens(a,b,aliens_pos,D):
    x1, y1 = a
    x2, y2 = b

    # Calculate Manhattan Distance
    distance = abs(x1 - x2) + abs(y1 - y2)

    # Calculate the minimum distance to any alien
    min_distance_to_alien = float('inf')
    for alien in aliens_pos:
        alien_x, alien_y = alien
        distance_to_alien = abs(x2 - alien_x) + abs(y2 - alien_y)
        min_distance_to_alien = min(min_distance_to_alien, distance_to_alien)

    # Penalize cells based on the minimum distance to any alien
    if min_distance_to_alien != 0:
        distance += D/min_distance_to_alien

    return distance

def bot_4_path_a_star(ship, D, start, goal, aliens_pos):
    fringe = PriorityQueue()
    fringe.put((0,start))
    dist = { start:0 }
    prev = {}
    while not fringe.empty():
        _, curr = fringe.get()
        if curr == goal :
            path = []
            while curr in prev:
                path.append(curr)
                curr = prev[curr]
            path.append(start)
            return path[::-1]

        x,y = curr
        neighbors = check_open_neighbours(ship, D, x, y)
        for neighbor in neighbors:
            tempDist = dist[curr] + 1
            if neighbor not in dist or tempDist < dist[neighbor] :
                dist[neighbor] = tempDist
                prev[neighbor] = curr
                priority = dist[neighbor] +heuristic_avoiding_aliens(neighbor,goal,aliens_pos,D)
                fringe.put((priority,neighbor))
    return None

def simulate_bot4(ship,D,start,goal,aliens_pos):

    i=0
    bot_idle_time = 0
    final_bot_path = [start]

    # Constraining the loop to a maximum of 1000 steps
    while i<1000:

        # Visualize the ship grid
        visualize_ship(ship,D,goal,status,final_bot_path)
        plt.pause(1)

        # Add the surrounding cells of aliens also as blocked cells
        buffer = get_alien_neighbours(ship, D, aliens_pos)
        new_ship = [row[:] for row in ship]
        for cell in buffer:
            x,y = cell
            new_ship[x][y] = '0'

        # Finding the path from the bot to the captain for the current ship configuration at each iteration
        bot_path = bot_4_path_a_star(new_ship, D, start , goal, aliens_pos)

        b_x,b_y = start
        new_x,new_y = start

        if(bot_path):

           # Move the bot
            new_x,new_y = bot_path[1]
            ship[b_x][b_y] = '1'
            ship[new_x][new_y] = 'B'
            final_bot_path.append((new_x,new_y))

        else:

            # The bot moves randomly, if the bot is idle for longer time.
            if(bot_idle_time >=5):
              s_x,s_y = start
              bot_neighbours = check_open_neighbours(new_ship,D,new_x,new_y)
              if(len(bot_neighbours) >= 1):
                  ship[s_x][s_y] = '1'
                  index = random.randint(0, len(bot_neighbours)-1)
                  new_x,new_y = bot_neighbours[index]
                  ship[new_x][new_y] = 'B'
                  final_bot_path.append((new_x,new_y))
                  bot_idle_time = 0
            else:
              bot_idle_time+=1


        # Check if bot gets attacked by aliens and then move the aliens
        status,aliens_pos = check_after_bot_moves(ship,D,goal,(new_x,new_y),aliens_pos)
        i+=1
        if status == "NEXT":
            start = (new_x,new_y)

            continue

        elif status in ("SUCCESS","FAILURE"):
            visualize_ship(ship,D,goal,status,final_bot_path)
            plt.pause(5)
            return status, i


    return None, i

"""# Main Function"""

def main():
    D = 20
    aliens_size = 20

    # Ship Layout Generation
    ship = generate_ship_layout(D)
    ship_copy = copy.deepcopy(ship)

    # Placing the bot randomly in an open cell
    open_cells = get_open_cells(ship_copy, D)
    index = random.randint(0, len(open_cells) - 1)
    b_x, b_y = open_cells[index]
    ship_copy[b_x][b_y] = 'B'

    aliens_pos = []


    # Placing the aliens randomly other than bot position in an open cell
    while True:
        index = random.randint(0, len(open_cells) - 1)
        new_x, new_y = open_cells[index]
        if ship_copy[new_x][new_y] != 'B':
            ship_copy[new_x][new_y] = 'A'
            aliens_size -= 1
            aliens_pos.append((new_x, new_y))
        if aliens_size == 0:
            break

    # Placing the captain randomly other than bot position in an open cell
    while True:
        index = random.randint(0, len(open_cells) - 1)
        c_x, c_y = open_cells[index]
        if ship_copy[c_x][c_y] != 'B':
            ship_copy[c_x][c_y] = 'C'
            break

    # Calculating success and survival efficiencies for all the bots
    # Uncomment the respective bot execution you want

    # Bot 1
    #status_bot, steps_bot = simulate_bot1(ship_copy, D, (b_x, b_y), (c_x, c_y), aliens_pos)


    # Bot 2
    #status_bot, steps_bot = simulate_bot2(copy.deepcopy(ship_copy), D, (b_x, b_y), (c_x, c_y), copy.deepcopy(aliens_pos))


    # Bot 3
    status_bot, steps_bot = simulate_bot3(copy.deepcopy(ship_copy), D, (b_x, b_y), (c_x, c_y), copy.deepcopy(aliens_pos))


    # Bot 4
    #status_bot, steps_bot = simulate_bot4(copy.deepcopy(ship_copy), D, (b_x, b_y), (c_x, c_y), copy.deepcopy(aliens_pos))

    print(status_bot," Steps taken - ", steps_bot)

"""# Testing"""

if __name__ == "__main__":
    main()
