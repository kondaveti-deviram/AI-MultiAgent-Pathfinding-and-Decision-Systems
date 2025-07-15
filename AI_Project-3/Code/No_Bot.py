import random
import numpy as np
import matplotlib.pyplot as plt


# Function to print the matrix
def print_matrix(matrix):
    for row in matrix:
        print(" ".join(map(str, row)))
     
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

    plt.imshow(visual_matrix, interpolation='nearest')
    plt.title('Ship Matrix Visualization')
    plt.grid(False)
    plt.show(block=False)
    plt.pause(0.2)


# Function to get open neighbours
def get_open_neighbours(ship, D, row, col):
    result = []
    if row - 1 >= 0 and ship[row - 1][col] not in ('0'):
        result.append((row-1,col))
    if row + 1 < D and ship[row + 1][col] not in ('0'):
        result.append((row+1,col))
    if col - 1 >= 0 and ship[row][col - 1] not in ('0'):
        result.append((row,col-1))
    if col + 1 < D and ship[row][col + 1] not in ('0'):
        result.append((row,col+1))
    return result

# Function to get open cells in the ship
def get_open_cells(ship, D):
    result = []
    for i in range(D):
        for j in range(D):
            if ship[i][j] in ('1'):
                result.append((i, j))
    return result

# Function to generate the ship
def generate_ship_layout(D):
    ship = [['1' for _ in range(D)] for _ in range(D)]

    T_row, T_col = D // 2, D // 2
    
    # Teleport
    ship[T_row][T_col] = 'T'
    
    # Immediate diagonal cells of the teleport are blocked cells
    if T_row - 1 >= 0 and T_col - 1 >= 0:
        ship[T_row - 1][T_row - 1] = '0'  # Top-left
    if T_row - 1 >= 0 and T_col + 1 < D:
        ship[T_row - 1][T_col + 1] = '0'  # Top-right
    if T_row + 1 < D and T_col - 1 >= 0:
        ship[T_row + 1][T_col - 1] = '0'  # Bottom-left
    if T_row + 1 < D and T_col + 1 < D:
        ship[T_row + 1][T_col + 1] = '0'  # Bottom-right
    
    teleport_neighbours = [(T_row - 1, T_col - 1), (T_row - 1, T_col + 1), 
                           (T_row + 1, T_col - 1), (T_row + 1, T_col + 1),
                           (T_row + 1, T_col), (T_row - 1, T_col),
                           (T_row, T_col - 1), (T_row, T_col + 1)]
    
    neighbours_list = []
    for cell in teleport_neighbours:
        (x, y) = cell
        neighbours = get_open_neighbours(ship, D, x, y)
        for neighbour in neighbours:
            neighbours_list.append(neighbour)
        
    blocked_count = 0
    random.seed() 
    while blocked_count < 10:
        row, col = random.randint(0, D - 1), random.randint(0, D - 1)
        if ship[row][col] == '1' and (row, col) not in neighbours_list and (row, col) not in teleport_neighbours:
            ship[row][col] = '0'
            blocked_count += 1
        
    return ship

# Function to check after the crew movement
def check_after_crew_moves(ship, D, crew_pos):

    T_row, T_col = D // 2, D // 2
    Teleport = (T_row, T_col )
    
    # If crew reaches teleport
    if(crew_pos == Teleport):
        #print("Success!!!")
        return "SUCCESS"

    return "NEXT"

# Function to calculate status and step for no bot scenario
def simulate_no_bot(ship, D, start):
    
    step = 1
    while True:
        
        c_x,c_y = start
        new_x,new_y = start
        
        # Move the captain randomly
        neighbours = get_open_neighbours(ship, D, c_x, c_y)
        next_move  = random.choice(neighbours)
        new_x,new_y = next_move
        ship[new_x][new_y] = 'C'
        ship[c_x][c_y] = '1'
        
        status = check_after_crew_moves(ship, D, start)
        step += 1
        
        if status == "NEXT": 
            start = (new_x,new_y)
            continue
        
        elif status == "SUCCESS":
            break
        
    return status,step
    
# Function to calculate the average steps from each position 
def get_simulated_crew_matrix(ship, D, teleport_pos, num_of_trials):
    calculate_steps = np.full((D, D), np.inf)
    
    for i in range(D):
        for j in range(D):
            if(ship[i][j] == '1'):
                
                start = (i,j)
                total_steps = 0
                for k in range(num_of_trials):
                    status,steps = simulate_no_bot(ship, D, start)
                    total_steps += steps
                
                average_steps = float(total_steps / num_of_trials)
                calculate_steps[i][j] = average_steps
    
    return calculate_steps
    
# Function to compute T_no_bot
def compute_T_no_bot(ship, D, teleport_pos):
    
    T_matrix = np.zeros((D,D))
    T_matrix[teleport_pos] = 0
    
    status_changed = True
    steps= 0
    while status_changed and steps < 10000:
        status_changed = False
        max_value = 0
        
        for i in range(D):
            for j in range(D):
                
                if (i,j) == teleport_pos or ship[i][j] == '0':
                    continue
                
                current_value = T_matrix[i,j]
                neighbors = get_open_neighbours(ship, D, i, j)
                if neighbors:
                    neighbor_values = []
                    for cell in neighbors:
                        neighbor_values.append(T_matrix[cell])
                    new_value = 1 + float(sum(neighbor_values) / len(neighbors))
                    T_matrix[i,j] = new_value
                    
                    # Check for significant change
                    if abs(new_value - current_value) > 0.01:  
                        status_changed = True
                        max_value = max(max_value, abs(new_value - current_value))
        steps += 1
        
    return T_matrix
                     
                  
def main():
    
    D = 11
    num_of_trials = 100
    
    #ship = generate_ship_layout(D)
    
    ship = [['1', '1', '0', '1', '1', '1', '1', '1', '1', '1', '1'], 
            ['1', '1', '1', '1', '1', '1', '0', '1', '1', '1', '0'], 
            ['0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1'], 
            ['1', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1'], 
            ['1', '1', '1', '1', '0', '1', '0', '1', '1', '1', '1'], 
            ['1', '1', '1', '1', '1', 'T', '1', '1', '1', '1', '1'], 
            ['1', '1', '1', '1', '0', '1', '0', '1', '1', '1', '1'], 
            ['1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '1'], 
            ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1'], 
            ['1', '1', '1', '1', '0', '1', '1', '0', '1', '0', '1'], 
            ['1', '1', '1', '1', '1', '0', '1', '1', '1', '1', '1']]

    # Teleport position
    teleport_pos = (D // 2 , D // 2)
    crew_simulated_matrix = get_simulated_crew_matrix(ship, D, teleport_pos, num_of_trials)
    
    print("Simulated Times Matrix:")
    print()
    print_matrix(crew_simulated_matrix)
    print()
    
    T_no_bot = compute_T_no_bot(ship, D, teleport_pos)
    print("T_no_Bot Matrix:")
    print()
    print_matrix(T_no_bot)
    print()

    for x in range(D):
        for y in range(D):
            print(x,y , '-->',T_no_bot[x][y])



if __name__ == "__main__":
    main()
