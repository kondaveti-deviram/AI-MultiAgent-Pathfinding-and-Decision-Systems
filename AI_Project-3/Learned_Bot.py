import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random
import copy
import matplotlib.pyplot as plt


# Read the data from csv file
def read_csv_file(csv_file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Extract data from specific columns into lists
    data = df.values.tolist()
    return data

# Path of the file to save the data
csv_file_path = "./data/t_bot_data_final.csv"
bot_optimal_moves = read_csv_file(csv_file_path)

# Extract the states and actions from the data
print("bot",bot_optimal_moves[0])
states = np.array([move[:4] for move in bot_optimal_moves])  # Bot and crew coordinates
actions = np.array([move[5:] for move in bot_optimal_moves])


# Split the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(states, actions, test_size=0.2, random_state=42)
print("Shape of training inputs:", x_train.shape)
print("Shape of testing inputs:", x_test.shape)
print("Shape of training outputs:", y_train.shape)
print("Shape of testing outputs:", y_test.shape)

# CNN Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),  # Input layer with 4 units
    Dense(128, activation='relu'),
    Dense(2,activation='linear')
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(states, actions, epochs=20, batch_size=32, validation_data=(x_test, y_test))


# Model Prediction
def predict_move(input):
    prediction = model.predict(input)[0]
    pred = []
    pred.append((int(np.round(prediction[0])),int(np.round(prediction[1]))))
    pred.append((int(prediction[0]),int(prediction[1])))
    print(prediction)
    return random.choice(pred)

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

# Generate Ship layout
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
        
    
    # Add 10 blocked cells to the ship
    blocked_count = 0
    random.seed() 
    while blocked_count < 10:
        row, col = random.randint(0, D - 1), random.randint(0, D - 1)
        if ship[row][col] == '1' and (row, col) not in neighbours_list and (row, col) not in teleport_neighbours:
            ship[row][col] = '0'
            blocked_count += 1

    return ship

def visualize_ship(ship, D):
    # Create a visual matrix representing the ship
    visual_matrix = np.zeros((D, D, 3), dtype=np.uint8)
    for i in range(D):
        for j in range(D):
            if ship[i][j] == '0':
                visual_matrix[i, j] = [0, 0, 0]  # Black for obstacles
            elif ship[i][j] == '1':
                visual_matrix[i, j] = [255, 255, 255]  # White for open cells
            elif ship[i][j] == 'T':
                visual_matrix[i, j] = [255, 0, 0]  # Red for teleport pad
            elif ship[i][j] == 'C':
                visual_matrix[i, j] = [0, 255, 0]  # Green for crew member
            elif ship[i][j] == 'B':
                visual_matrix[i, j] = [0, 0, 255]   # Blue for crew member

    plt.imshow(visual_matrix, interpolation='nearest')
    plt.title('Ship Matrix Visualization')
    plt.grid(False)
    plt.show(block=False)
    plt.pause(0.2)

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

# Function to get open cells in the ship
def get_open_cells(ship, D):
    result = []
    for i in range(D):
        for j in range(D):
            if ship[i][j] == '1':
                result.append((i, j))
    return result


# Bot and Crew Movement simulation

D = 11

org_ship =[
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

# Teleport pad position
teleport_pos = (D // 2 , D // 2)
steps_count = []

for i in range(10):
    ship = copy.deepcopy(org_ship)

    # Place a bot and crew
    bot_x=7
    bot_y=9
    ship[bot_x][bot_y]='B'
    crew_x=9
    crew_y=10
    ship[crew_x][crew_y]='C'
    step=0
    stay_count = 0
    while step<300:

        # Predict the bot movement from the model
        input = np.array([[bot_x, bot_y,crew_x, crew_y]])
        prediction = predict_move(input)
        optimal_move = (int(np.round(prediction[0])),int(np.round(prediction[1])))
        print(input,optimal_move)

        if(optimal_move == (bot_x,bot_y) or optimal_move == (crew_x,crew_y)):
            stay_count += 1
        else:
            stay_count = 0

        if(stay_count > 3):
            prediction = predict_move(input)
            optimal_move = (int(prediction[0]),int(prediction[1]))
            #optimal_move = random.choice(bot_neighbors)
        elif(stay_count > 5):
            bot_neighbors = get_bot_neighbours(ship,D,bot_x,bot_y)
            optimal_move = random.choice(bot_neighbors)


        # Move the bot to the posiition predicted by model
        if 0<=optimal_move[0]<D and 0<=optimal_move[1]<D and ship[optimal_move[0]][optimal_move[1]]!='0' and optimal_move != (crew_x,crew_y):
            ship[bot_x][bot_y] = '1'  # Clear old bot position
            bot_x, bot_y = optimal_move
            ship[bot_x][bot_y] = 'B'  # Set new bot position

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


