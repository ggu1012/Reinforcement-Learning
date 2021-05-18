from environment import grid_world
from agent import AGENT


WORLD_HEIGHT = 5
WORLD_WIDTH = 10

env = grid_world(WORLD_HEIGHT,WORLD_WIDTH,
                 GOAL = [[WORLD_HEIGHT-1, WORLD_WIDTH-1]],
                 OBSTACLES=[[0,2], [1,2], [2,2], [0,4], [2,4], [4,4], [2, 6],
                            [3, 6],[4, 6], [2,7], [2,8]])

agent = AGENT(env,is_upload=False)
agent.Q_learning(epsilon=0.4,decay_period=10000, decay_rate=0.8)





