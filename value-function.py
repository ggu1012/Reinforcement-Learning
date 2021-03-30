import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

HEIGHT = 4
WIDTH = 4

class grid_world:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT

    def is_terminal(self, state):   # Gaol state
        x, y = state
        return (x == 0 and y == 0) or (x == self.width - 1 and y == self.height - 1)

    def interaction(self, state, action):
        if self.is_terminal(state):
            return state, 0

        next_state = (np.array(state) + action).tolist()
        x, y = next_state

        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            next_state = state

        reward = -1
        return next_state, reward

    def size(self):
        return self.width, self.height


def draw_image(iteration, image):
    fig, ax = plt.subplots()
    plt.suptitle('Iteration:{:d}'.format(iteration))
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), val in np.ndenumerate(image):
        tb.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor='white')

    # Row and column labels...
    for i in range(len(image)):
        tb.add_cell(i, -1, width, height, text=i+1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height/2, text=i+1, loc='center',
                    edgecolor='none', facecolor='none')
    ax.add_table(tb)

    plt.show()


WORLD_SIZE = 4
# left, up, right, down
ACTIONS = {'LEFT': np.array([0, -1]), 'UP': np.array([-1, 0]),
           'RIGHT': np.array([0, 1]), 'DOWM': np.array([1, 0])}
ACTION_PROB = 0.25


def evaluate_state_value_by_matrix_inversion(env, discount=1.0):
    WIDTH, HEIGHT = env.size()

    # Reward matrix R
    R = np.zeros((WIDTH, HEIGHT))
    for i in range(WIDTH):
        for j in range(HEIGHT):
            expected_reward = 0
            for action in ACTIONS:
                (next_i, next_j), reward = env.interaction(
                    [i, j], ACTIONS[action])
                expected_reward += ACTION_PROB*reward
            R[i, j] = expected_reward
    R = R.reshape((-1, 1))
    R = R[1:-1, :]

    # Transition matrix T

    # T has an initial size of 4x4x4x4
    T = np.zeros([WIDTH,  HEIGHT, WIDTH, HEIGHT])

    # This interaction code lines are identical to the ones
    # used in the reward interactions.
    # After this loop, T array would contain the information
    # of the probability of each cell's movement.

    # Horizontal movement
    for i in range(WIDTH):
        # Vertical movement
        for j in range(HEIGHT):
            # Control interaction
            for action in ACTIONS:

                # next_i and next_i components are the key part
                # of the control interaction. This line computes
                # P(s'|s,a) of each cells.
                (next_i, next_j), reward = env.interaction(
                    [i, j], ACTIONS[action])

                # '+=' operator is used to consider the case
                # when the cell tried to move but it couldn't
                # because of the wall. In this case, S' would
                # be same as S.
                T[i, j, next_i, next_j] += ACTION_PROB



    # Flattens T matrix into [-1,] shape.
    # As a result, T becomes 256x1 array.
    T = T.flatten()

    # Reshape flattened matrix into [16,16]
    # Now, the information of the probability is expressed as T[s',s].
    T = T.reshape([WIDTH*HEIGHT,WIDTH*HEIGHT])

    # Since T has contained the information of terminal state 1 and 16,
    # those information are truncated in this line.
    T = T[1:-1, 1:-1]

    
    # iteration variable for debugging
    iteration = 1000     


    # Initial V matrix = 0s
    V = np.zeros([14,1])
    
    # Two expressions for finding V

    # 1. V = R + discount*T*V
    # With the sufficient iterations, 
    # the result becomes identical to the 2. case.
    for i in range(iteration):
        V = R + np.matmul(T, V)
    
    # 2. V = inv(I-T)*R
    # Identity matrix for I
    I = np.identity(WIDTH*HEIGHT-2)
    # V = inv(I-T)*R
    V = np.matmul(np.linalg.inv(I-T), R)

    # Now, V has a shape of [14,1]
    
    # This line flattens the array into [-1,] shape for concatenation.
    V = V.flatten()

    # Inserts terminal state. 
    # Then, V contains full information of the grid world.
    V = np.concatenate(([0], V, [0]))



    new_state_values = V.reshape(WIDTH, HEIGHT)
    draw_image(iteration, np.round(new_state_values, decimals=2))

    return new_state_values


env = grid_world()
values = evaluate_state_value_by_matrix_inversion(env=env)
