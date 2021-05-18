import numpy as np
from visualize_train import draw_value_image, draw_policy_image

# left, right, up, down
ACTIONS = [np.array([0, -1]),
           np.array([0, 1]),
           np.array([-1, 0]),
           np.array([1, 0])]

TRAINING_EPOCH_NUM = 500000

class AGENT:
    def __init__(self, env, is_upload=False):

        self.ACTIONS = ACTIONS
        self.env = env
        HEIGHT, WIDTH = env.size()
        self.state = [0,0]

        if is_upload:
            qlearning_results = np.load('./result/qlearning.npz')
            self.V_values = qlearning_results['V']
            self.Q_values = qlearning_results['Q']
            self.policy = qlearning_results['PI']
        else:
            self.V_values = np.zeros((HEIGHT, WIDTH))
            self.Q_values = np.zeros((HEIGHT, WIDTH, len(self.ACTIONS)))
            self.policy = np.zeros((HEIGHT, WIDTH,len(self.ACTIONS)))+1./len(self.ACTIONS)


    def initialize_episode(self):
        HEIGHT, WIDTH = self.env.size()
        while True:
            i = np.random.randint(HEIGHT)
            j = np.random.randint(WIDTH)
            state = [i, j]
            if (state in self.env.goal) or (state in self.env.obstacles):
                continue
            break
            # if (state not in self.env.goal) and (state not in self.env.obstacles):
            #     break
        return state


    def policy_summary(self, epsilon):
        HEIGHT, WIDTH = self.env.size()
        self.policy = np.zeros((HEIGHT, WIDTH, len(self.ACTIONS))) + 1. / len(self.ACTIONS)
        for i in range(HEIGHT):
            for j in range(WIDTH):
                self.policy[i, j, :] = np.zeros(len(ACTIONS))
                greedy_action_index = np.argmax(self.Q_values[i, j, :])
                self.policy[i, j, greedy_action_index] = (1. - epsilon)
                self.policy[i, j, :] += epsilon / len(ACTIONS)



    def Q_learning(self, discount=1.0, alpha=0.01, max_seq_len=500,
                            epsilon=0.3, decay_period=20000, decay_rate=0.9):

        for episode in range(TRAINING_EPOCH_NUM):
            state = self.initialize_episode()
            done = False
            timeout = False
            seq_len = 0

            while not (done or timeout):
                # Next state and action generation
                action = self.get_action(state, epsilon)
                movement = ACTIONS[action]
                next_state, reward = self.env.interaction(state, movement)                

                # ***********   Q value update   ****************

                # Find arg max_a Q(s',a)
                next_i, next_j = next_state
                max_idx = np.argmax(self.Q_values[next_i, next_j, :])
                target = self.Q_values[next_i, next_j, max_idx]
                
                i, j = state
                self.Q_values[i, j, action] += alpha * (reward + discount * target - self.Q_values[i, j, action])

                state = next_state

                # ************************************************

                seq_len += 1
                if (seq_len >= max_seq_len):
                    timeout = True
                done = self.env.is_terminal(state)

            if episode % 10000 == 0:
                print("Num of episodes = {:}, epsilon={:.4f}".format(episode, epsilon))

            if episode % decay_period == 0:
                epsilon *= decay_rate

        self.V_values = np.max(self.Q_values, axis=2)
        draw_value_image(1, np.round(self.V_values, decimals=2), env=self.env)
        self.policy_summary(epsilon)
        draw_policy_image(1, np.round(self.policy, decimals=2), env=self.env)
        np.savez('./result/qlearning.npz', Q=self.Q_values, V=self.V_values, PI=self.policy)
        return self.Q_values, self.V_values, self.policy


    def get_action(self, state, epsilon):  # epsilon-greedy
        i, j = state
        if np.random.rand() < epsilon:
            action = np.random.choice(len(ACTIONS))
        else:
            action = np.argmax(self.Q_values[i, j, :])
        return action
