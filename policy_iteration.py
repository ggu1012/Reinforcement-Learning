# visualize_train
import matplotlib.pyplot as plt
from matplotlib.table import Table
import numpy as np
import matplotlib

ACTION_SYMBOLS = {0: '←', 1: '→', 2: '↑', 3: '↓'}


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
                actions = (np.where(policy_image[i, j, :] != 0)[0]).tolist()
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
        self.state = [0, 0]

        if is_upload:
            self.values = values
            self.policy = policy
        else:
            self.values = np.zeros((HEIGHT, WIDTH))
            self.policy = np.zeros(
                (HEIGHT, WIDTH, len(self.ACTIONS)))+1./len(self.ACTIONS)

    # policy_evaluation part is almost identical to the previous homework.

    def policy_evaluation(self, iter, env, policy, discount=1.0):
        HEIGHT, WIDTH = env.size()
        new_state_values = np.zeros((HEIGHT, WIDTH))
        iteration = 1

        # Reward matrix R
        R = np.zeros((HEIGHT, WIDTH))

        # Transition matrix T
        # T has an initial size of 5x10x5x10
        T = np.zeros([HEIGHT, WIDTH, HEIGHT, WIDTH])

        # For each state [i,j],
        for i in range(HEIGHT):
            for j in range(WIDTH):
                expected_reward = 0
                # For each action,
                for k in range(len(ACTIONS)):
                    # Environment interaction.
                    # Computes the next state and the reward following the action.
                    (next_i, next_j), reward = env.interaction([i, j], ACTIONS[k])

                    # policy[i,j,k] : probability of ACTIONS[k] for state [i,j]
                    # Trainstion matrix T would contain the information of the probability
                    # of each cell's movement considering obstacles and walls.
                    expected_reward += self.policy[i, j, k] * reward
                    T[i, j, next_i, next_j] += self.policy[i, j, k]

                R[i, j] = expected_reward

        # Below this lines are identical to the ones used in previous HW
        # except the size of the gridworld.

        R = R.reshape((-1, 1))
        R = R[1:-1, :]

        # Flattens T matrix into [-1,] shape.
        # As a result, T becomes 2500x1 array.
        T = T.flatten()

        # Reshape flattened matrix into [50,50]
        # Now, the information of the probability is expressed as T[s,s'].
        T = T.reshape([WIDTH*HEIGHT, WIDTH*HEIGHT])

        # Since T has contained the information of terminal state 1 and 50,
        # those information are truncated in this line.
        T = T[1:-1, 1:-1]

        # Initial V matrix = 0s
        V = np.zeros([HEIGHT*WIDTH - 2, 1])

        # V = inv(I-T)*R
        # Identity matrix for I
        I = np.identity(WIDTH*HEIGHT-2)

        # V = inv(I-T)*R
        V = np.matmul(np.linalg.inv(I-T), R)

        # This line flattens the array into [-1,] shape for concatenation.
        V = V.flatten()

        # Inserts terminal state.
        # Then, V contains full information of the grid world.
        V = np.concatenate(([0], V, [0]))

        new_state_values = V.reshape(HEIGHT, WIDTH)

        draw_value_image(iter, np.round(new_state_values, decimals=2), env=env)
        return new_state_values, iteration

    def policy_improvement(self, iter, env, state_values, old_policy, discount=1.0):
        HEIGHT, WIDTH = env.size()
        policy = old_policy.copy()

        # Based on the policy improvement theorem,
        #
        # 1. Take one random action out of 4 actions.
        #
        # 2. Compute the instant reward and add the result with the state value
        #    obtained from the policy evaluation.
        #    Try for each action and save the result.
        #
        # 3. Compare the 4 results. 
        #    Choose the largest result. If there is more than one, take that as well.
        #    Append the indices of the chosens to the list.
        #
        # 4. Update the policy.
        #    ACTIONS[index] = 1/(length of the indices list)
        #    ACTIONS[otherwise] = 0

        # For every states
        for i in range(HEIGHT):
            for j in range(WIDTH):
                return_value = np.zeros([len(ACTIONS)])
                new_policy_state = np.array(np.zeros([1, 4]), dtype=float).flatten()

                # For every action,
                for k in range(len(ACTIONS)):
                    action = ACTIONS[k]
                    # Computes the next state and the reward following the action.
                    [next_i, next_j], instant_reward = env.interaction([i, j], action)

                    # Compute the instant reward and add the result with the state value.             
                    return_value[k] = instant_reward + state_values[next_i, next_j]

                # Choose the largest result. If there is more than one, take that as well.
                # Append the indices of the chosens to the list.
                max_return_idx = np.argwhere(return_value == np.max(return_value))

                # Convert np.array to list
                max_return_idx = max_return_idx.flatten().tolist()

                # Update the policy based on the written procedure above.
                for index in max_return_idx:
                    new_policy_state[index] = 1/len(max_return_idx)
                policy[i, j] = new_policy_state

        # If the updated policy is identical to the old policy,
        # it means the policy is converged to the optimal policy.
        # In that case, return True.
        policy_stable = np.all(np.abs(policy - old_policy) == 0)

        print('policy stable {}:'.format(policy_stable))
        draw_policy_image(iter, np.round(policy, decimals=2), env=env)
        return policy, policy_stable

    def policy_iteration(self):
        iter = 1
        while (True):
            self.values, iteration = self.policy_evaluation(
                iter, env=self.env, policy=self.policy)
            self.policy, policy_stable = self.policy_improvement(
                iter, env=self.env, state_values=self.values, old_policy=self.policy, discount=1.0)
            iter += 1
            if policy_stable == True:
                break
        return self.values, self.policy

    def get_action(self, state):
        i, j = state
        return np.random.choice(len(ACTIONS), 1, p=self.policy[i, j, :].tolist()).item()

    def get_state(self):
        return self.state


# environment

class grid_world:

    def __init__(self, HEIGHT, WIDTH, GOAL, OBSTACLES):
        self.height = HEIGHT
        self.width = WIDTH
        self.goal = GOAL
        self.obstacles = OBSTACLES

    def is_terminal(self, state):   # Goal state
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

        r = self.reward(state, action, next_state)
        return next_state, r

    def size(self):
        return self.height, self.width

# main_train


WORLD_HEIGHT = 5
WORLD_WIDTH = 10

env = grid_world(WORLD_HEIGHT, WORLD_WIDTH,
                 GOAL=[[0, 0], [WORLD_HEIGHT-1, WORLD_WIDTH-1]],
                 OBSTACLES=[[2, 5], [1, 2], [2, 2], [3, 2], [2, 8], [3, 8], [4, 8]])
agent = AGENT(env, is_upload=False)
values, policy = agent.policy_iteration()
