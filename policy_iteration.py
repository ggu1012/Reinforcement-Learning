# visualize_train
import matplotlib.pyplot as plt
from matplotlib.table import Table
import numpy as np
import matplotlib

ACTION_SYMBOLS = {0:'←', 1:'→', 2:'↑', 3:'↓'}

def draw_value_image(iteration, value_image, env):
    fig, ax = plt.subplots()
    plt.suptitle('Policy Evaluation: Iteration:{:d}'.format(iteration))
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = value_image.shape
    height, width = 1.0 / nrows, 1.0 / ncols

    # Add cells
    for (i, j), val in np.ndenumerate(value_image):
        if env.is_terminal([i, j]):
            tb.add_cell(i, j, height, width, text=' ',
                        loc='center', facecolor='white')
        elif env.is_on_obstacle([i, j]):
            tb.add_cell(i, j, height, width, text='╳',
                        loc='center', facecolor='white')
        else:
            tb.add_cell(i, j, height, width, text=val,
                        loc='center', facecolor='white')

    # Row and column labels...
    for i in range(nrows):
        tb.add_cell(i, -1, height, width, text=i+1, loc='right',
                    edgecolor='none', facecolor='none')
    for i in range(ncols):
        tb.add_cell(nrows, i, height, width/2, text=i+1, loc='center',
                    edgecolor='none', facecolor='none')
    ax.add_table(tb)
    plt.show()



def draw_policy_image(iteration, policy_image, env):
    fig, ax = plt.subplots()
    plt.suptitle('Policy Improvement: Iteration:{:d}'.format(iteration))
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols, nactinos = policy_image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for i in range(nrows):
        for j in range(ncols):
            if env.is_terminal([i, j]):
                tb.add_cell(i, j, height, width, text=' ',
                        loc='center', facecolor='white')
            elif env.is_on_obstacle([i, j]):
                tb.add_cell(i, j, height, width, text='╳',
                        loc='center', facecolor='white')
            else:
                actions = (np.where(policy_image[i,j,:] != 0)[0]).tolist()
                actions_text = ''.join(ACTION_SYMBOLS[x] for x in actions)
                tb.add_cell(i, j, height, width, text=actions_text,
                        loc='center', facecolor='white')

    # Row and column labels...
    for i in range(nrows):
        tb.add_cell(i, -1, height, width, text=i+1, loc='right',
                    edgecolor='none', facecolor='none')
    for i in range(ncols):
        tb.add_cell(nrows, i, height, width/2, text=i+1, loc='center',
                    edgecolor='none', facecolor='none')
    ax.add_table(tb)

    plt.show()


# agent
values = 0
policy = 0

ACTIONS = [np.array([0, -1]),
           np.array([0, 1]),
           np.array([-1, 0]),
           np.array([1, 0])]


class AGENT:
    def __init__(self, env, is_upload=False):

        self.ACTIONS = ACTIONS
        self.env = env
        HEIGHT, WIDTH = env.size()
        self.state = [0,0]

        if is_upload:
            self.values = values
            self.policy = policy
        else:
            self.values = np.zeros((HEIGHT, WIDTH))
            self.policy = np.zeros((HEIGHT, WIDTH,len(self.ACTIONS)))+1./len(self.ACTIONS)


    def policy_evaluation(self, iter, env, policy, discount=1.0):
        HEIGHT, WIDTH = env.size()
        new_state_values = np.zeros((HEIGHT, WIDTH))
        iteration = 0

        #***************************************************
        #
        #
        #
        #         Write your code down
        #
        #
        #
        #
        #***************************************************

        draw_value_image(iter, np.round(new_state_values, decimals=2), env=env)
        return new_state_values, iteration


    def policy_improvement(self, iter, env, state_values, old_policy, discount=1.0):
        HEIGHT, WIDTH = env.size()
        policy = old_policy.copy()

        #***************************************************
        #
        #
        #
        #         Write your code down
        #
        #
        #
        #
        #***************************************************

        print('policy stable {}:'.format(policy_stable))
        draw_policy_image(iter, np.round(policy, decimals=2), env=env)
        return policy, policy_stable



    def policy_iteration(self):
        iter = 1
        while (True):
            self.values, iteration = self.policy_evaluation(iter, env=self.env, policy=self.policy)
            self.policy, policy_stable = self.policy_improvement(iter, env=self.env, state_values=self.values,
                                                       old_policy=self.policy, discount=1.0)
            iter += 1
            if policy_stable == True:
                break
        return self.values, self.policy



    def get_action(self, state):
        i,j = state
        return np.random.choice(len(ACTIONS), 1, p=self.policy[i,j,:].tolist()).item()


    def get_state(self):
        return self.state


# environment

class grid_world:

    def __init__(self, HEIGHT, WIDTH, GOAL, OBSTACLES):
        self.height = HEIGHT
        self.width = WIDTH
        self.goal = GOAL
        self.obstacles = OBSTACLES

    def is_terminal(self, state):   # Gaol state
        return state in self.goal

    def is_out_of_boundary(self, state):
        x, y = state
        if x < 0 or x >= self.height or y < 0 or y >= self.width:
            return True
        else:
            return False

    def is_on_obstacle(self, state):
        if state in self.obstacles:
            return True
        else:
            return False

    def reward(self, state, action, next_state):
        if self.is_terminal(state):
            return 0
        else:
            return -1

    def interaction(self, state, action):
        if self.is_terminal(state):
            next_state = state
        else:
            next_state = (np.array(state) + action).tolist()

        if self.is_out_of_boundary(next_state):
            next_state = state

        if self.is_on_obstacle(next_state):
            next_state = state

        r = self.reward(state,action,next_state)
        return next_state, r

    def size(self):
        return self.height, self.width
