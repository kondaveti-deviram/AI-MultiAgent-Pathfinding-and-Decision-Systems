import random
import numpy as np
import matplotlib.pyplot as plt
import csv
import copy

# Function to visualize the ship
def visualize_ship(ship, D):
    # Create a visual matrix representing the ship
    visual_matrix = np.zeros((D, D, 3), dtype=np.uint8)
    for i in range(D):
        for j in range(D):
            if ship[i][j] == '0':
                visual_matrix[i, j] = [0, 0, 0]        # Black for obstacles
            elif ship[i][j] == '1':
                visual_matrix[i, j] = [255, 255, 255]  # White for open cells
            elif ship[i][j] == 'T':
                visual_matrix[i, j] = [255, 0, 0]      # Red for teleport pad
            elif ship[i][j] == 'C':
                visual_matrix[i, j] = [0, 255, 0]      # Green for crew member
            elif ship[i][j] == 'B':
                visual_matrix[i, j] = [0, 0, 255]      # Blue for crew member

    plt.imshow(visual_matrix, interpolation='nearest')
    plt.title('Ship Matrix Visualization')
    plt.grid(False)
    plt.show(block=False)
    plt.pause(0.2)

# Function to write data into a csv file
def csv_writer(csv_file_path,data,actions):
    
    # Define the fieldnames (column names) for the CSV file
    fieldnames = ["bx","by", "cx","cy","t_bot","action_x","action_y"]
    
    # Write epoch data to the CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write header (fieldnames)
        writer.writeheader()
        t_bot_data = {}
        # Write data rows
        for state in data:
            bot,crew = state
            t_bot_data["bx"] = bot[0]
            t_bot_data["by"] = bot[1]
            t_bot_data["cx"] = crew[0]
            t_bot_data["cy"] = crew[1]
            t_bot_data["t_bot"] = data[state]
            t_bot_data["action_x"], t_bot_data["action_y"] = actions[state]
            writer.writerow(t_bot_data)

# Function to get open cells in the ship
def get_open_cells(ship, D):
    result = []
    for i in range(D):
        for j in range(D):
            if ship[i][j] in ('1'):
                result.append((i, j))
    return result

# Function for getting open crew neighbours
# CREW DIRECTIONS - UP, DOWN, LEFT, RIGHT
def get_crew_neighbours(ship, D, row, col):
    result = []
    if row - 1 >= 0 and ship[row - 1][col] not in ('0','B'):
        result.append((row-1,col))
    if row + 1 < D and ship[row + 1][col] not in ('0','B'):
        result.append((row+1,col))
    if col - 1 >= 0 and ship[row][col - 1] not in ('0','B'):
        result.append((row,col-1))
    if col + 1 < D and ship[row][col + 1] not in ('0','B'):
        result.append((row,col+1))
    return result

# Function for getting open bot neighbours 
# BOT DIRECTIONS - UP, DOWN, LEFT, RIGHT,UP-LEFT, UP-RIGHT, DOWN-LEFT, DOWN-RIGHT
def get_bot_neighbours(ship, D, row, col):
    result = []
    if row - 1 >= 0 and ship[row - 1][col] not in ('0','C'):
        result.append((row-1,col))
    if row + 1 < D and ship[row + 1][col] not in ('0','C'):
        result.append((row+1,col))
    if col - 1 >= 0 and ship[row][col - 1] not in ('0','C'):
        result.append((row,col-1))
    if col + 1 < D and ship[row][col + 1] not in ('0','C'):
        result.append((row,col+1))
    if row - 1 >= 0 and col - 1 >=0 and ship[row - 1][col-1] not in ('0','C'):
        result.append((row-1,col-1))
    if row + 1 < D and col + 1 < D and ship[row + 1][col+1] not in ('0','C'):
        result.append((row+1,col+1))
    if col - 1 >= 0 and row + 1 < D and ship[row+1][col - 1] not in ('0','C'):
        result.append((row+1,col-1))
    if col + 1 < D and row - 1 >= 0 and ship[row-1][col + 1] not in ('0','C'):
        result.append((row-1,col+1))
    return result
    
# Function compute initial time taken to reach teleport for each cell
def compute_T_bot(ship_conf, D, teleport_pos):
    T_bot = {}
    for i in range(D):
        for j in range(D):
            for k in range(D):
                for l in range(D):
                    T_bot[(i,j),(k,l)] = 9999
                    if (k,l) == teleport_pos:
                        T_bot[(i,j),(k,l)] = 0
    
    status_changed = True
    steps= 0
    
    # Iterate until convergence
    while status_changed:
        
        # Create a copy of the value function for comparison
        status_changed = False
        ship = copy.deepcopy(ship_conf) 
        
        # Update the value function for each state using the Bellman update equation
        for bx in range(D):
            for by in range(D):
                for cx in range(D):
                    for cy in range(D):
                        if((bx,by) != (cx,cy) and ship[bx][by] != '0' and ship[cx][cy] != '0'):
                            ship[bx][by] = 'B' 
                            ship[cx][cy] = 'C' 
                            if (cx,cy) == teleport_pos:
                                continue
                            current_value = T_bot[(bx,by),(cx,cy)]
                            bot_action = (bx,by)
                            min_value = 9999
                            actions = get_bot_neighbours(ship, D, bx, by)
                            ship[bx][by] ='1'
                            # Check for each bot movement
                            for bot_next in actions:
                                ship[bot_next[0]][bot_next[1]] = 'B'
                                neighbors = get_crew_neighbours(ship,D,cx,cy)
                                sum=0
                                if(bot_next in neighbors):
                                    neighbors.remove(bot_next)
                                
                                # Check for each neighbor of the crew
                                for crew_next in neighbors:
                                    sum += 1/len(neighbors) * T_bot[bot_next,crew_next]
                                value = 1 + sum

                                if(value<min_value ):
                                    min_value = value
                                    T_bot[(bx,by),(cx,cy)] = min_value

                                ship[bot_next[0]][bot_next[1]] = '1'
                            ship[bx][by] = ship_conf[bx][by] 
                            ship[cx][cy] = ship_conf[cx][cy] 
                            if abs(T_bot[(bx,by),(cx,cy)] - current_value) > 0.01:  # Check for significant change, else converge
                                status_changed = True
        steps += 1 
    return T_bot

# Function for T bot simulationa and finding optimal bot movement for each state 
def compute_optimal_policy(ship_conf, D, teleport_pos, T_bot_timestamp):
    T_bot = {}
    T_bot_action = {}
    for i in range(D):
        for j in range(D):
            for k in range(D):
                for l in range(D):
                    T_bot[(i,j),(k,l)] = 9999
                    T_bot_action[(i,j),(k,l)] = (i,j)
                    if (k,l) == teleport_pos:
                        T_bot[(i,j),(k,l)] = 0
    
    status_changed = True
    steps= 0
    
    # Iterate until convergence
    while status_changed:
        
        # Create a copy of the value function for comparison
        status_changed = False
        max_value = 0
        ship = copy.deepcopy(ship_conf)
        
        # Update the value function for each state using the Bellman update equation
        for bx in range(D):
            for by in range(D):
                for cx in range(D):
                    for cy in range(D):

                        if((bx,by) != (cx,cy) and ship[bx][by] != '0' and ship[cx][cy] != '0'):
                            ship[bx][by] = 'B' 
                            ship[cx][cy] = 'C' 
                            if (cx,cy) == teleport_pos:
                                continue
                            current_value = T_bot[(bx,by),(cx,cy)]
                            bot_action = (bx,by)
                            max_value = 9999
                            actions = get_bot_neighbours(ship, D, bx, by)

                            ship[bx][by] ='1'
                            
                            # Check for each bot movement
                            for bot_next in actions:
                                ship[bot_next[0]][bot_next[1]] = 'B'
                                #reward = calculate_reward(D,bot_next[0],bot_next[1],cx,cy,T_bot_timestamp)
                                reward = T_bot_timestamp[(bot_next[0],bot_next[1]),(cx,cy)]
                                neighbors = get_crew_neighbours(ship,D,cx,cy)
                                neighbors.append((cx,cy))
                                sum=0
                                
                                # Check for each neighbor of the crew
                                for crew_next in neighbors:
                                    sum +=  1/len(neighbors) * T_bot[bot_next,crew_next]
                                value = reward + sum
                                
                                if(value<max_value ):
                                    max_value = value
                                    bot_action = bot_next
                                    T_bot[(bx,by),(cx,cy)] = max_value
                                    T_bot_action[(bx,by),(cx,cy)] = bot_action

                                ship[bot_next[0]][bot_next[1]] = '1'
                            ship[bx][by] = ship_conf[bx][by] 
                            ship[cx][cy] = ship_conf[cx][cy]
                            if abs(T_bot[(bx,by),(cx,cy)] - current_value) > 0.01:
                                status_changed = True
        steps += 1
    return T_bot, T_bot_action      
    
def main():
    D = 11
    csv_file_path = "./t_bot_data_final.csv"
    #ship = generate_ship_layout(D)

    org_ship = [
        ['1', '1', '1', '1', '1', '1', '0', '1', '1', '1', '1'],
        ['1', '1', '1', '0', '1', '1', '1', '1', '0', '1', '1'], 
        ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1'], 
        ['1', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1'], 
        ['1', '1', '1', '1', '0', '1', '0', '1', '1', '0', '1'], 
        ['1', '1', '1', '1', '1', 'T', '1', '1', '1', '1', '1'], 
        ['1', '0', '1', '1', '0', '1', '0', '1', '1', '1', '1'], 
        ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1'], 
        ['1', '1', '0', '1', '1', '1', '1', '1', '1', '1', '0'], 
        ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1'], 
        ['1', '1', '1', '1', '0', '1', '1', '1', '0', '1', '1']
    ]
    
    teleport_pos = (D // 2 , D // 2)
    t_bot_timestamp = compute_T_bot(org_ship, D, teleport_pos)
        
    T_bot, actions = compute_optimal_policy(org_ship, D, teleport_pos,t_bot_timestamp)
    bot_x=9
    bot_y=9
    org_ship[bot_x][bot_y]='B'
    crew_x=9
    crew_y=10
    org_ship[crew_x][crew_y]='C'
    csv_writer(csv_file_path,T_bot,actions)
    
    steps_count = []
    for i in range(1):
        ship = copy.deepcopy(org_ship)
        step=0
        while True:
            # Extract the bot movement from the optimal policy
            optimal_move = actions[(bot_x, bot_y), (crew_x, crew_y)]

            if optimal_move:
                ship[bot_x][bot_y] = '1' 
                bot_x, bot_y = optimal_move
                ship[bot_x][bot_y] = 'B'

            # Handle crew movement
            crew_neighbours = get_crew_neighbours(ship, D, crew_x, crew_y)
            if crew_neighbours:
                x_new, y_new = random.choice(crew_neighbours)
                ship[crew_x][crew_y] = '1' 
                crew_x, crew_y = x_new, y_new
                ship[crew_x][crew_y] = 'C' 
            
            step += 1  
            
            # Game over if crew reaches teleport pad
            if (crew_x, crew_y) == teleport_pos:
                print("Crew reached the teleport. Teleported after ", step, "steps.")
                break
        steps_count.append(step)

    print("Average Steps = ",sum(steps_count)/len(steps_count))


if __name__ == "__main__":
    main()




