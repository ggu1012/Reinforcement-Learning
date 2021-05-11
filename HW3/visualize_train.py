import matplotlib.pyplot as plt
from matplotlib.table import Table
import numpy as np

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



