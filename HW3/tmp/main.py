import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from environment import GraphicDisplay, grid_world
from matplotlib.table import Table

matplotlib.use('Agg')

WORLD_SIZE = 4
# left, up, right, down
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])
]
ACTION_PROB = 0.25




def draw_image(image):
    fig, ax = plt.subplots()
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


def compute_state_value(env, discount=1.0):
    WIDTH, HEIGHT = env.size()
    new_state_values = np.zeros((WIDTH, HEIGHT))
    iteration = 0
    # state_values = new_state_values.copy()

    while True:
    #     if in_place:
    #         state_values = new_state_values
    #     else:
        state_values = new_state_values.copy()
    #     old_state_values = state_values.copy()

        for i in range(WIDTH):
            for j in range(HEIGHT):
                value = 0
                for action in ACTIONS:
                    (next_i, next_j), reward = env.interaction([i, j], action)
                    value += ACTION_PROB * (reward + discount * state_values[next_i, next_j])
                new_state_values[i, j] = value

        max_delta_value = abs(state_values - new_state_values).max()
        if max_delta_value < 1e-4:
            break
        iteration += 1
        print('Iteration:{}, max_delta:{:.5f}'.format(iteration, max_delta_value))
    return new_state_values, iteration


def figure_4_1():
    # While the author suggests using in-place iterative policy evaluation,
    # Figure 4.1 actually uses out-of-place version.
    # _, asycn_iteration = compute_state_value(in_place=True)

    env = grid_world()
    values, sync_iteration = compute_state_value(env = env)
    draw_image(np.round(values, decimals=2))
    # print('In-place: {} iterations'.format(asycn_iteration))
    # print('Synchronous: {} iterations'.format(sync_iteration))
    plt.savefig('../images/figure_4_1.png')
    plt.close()
    grid_world_vis = GraphicDisplay()
    grid_world_vis.print_value_table(values)
    grid_world_vis.mainloop()


if __name__ == '__main__':
    figure_4_1()

