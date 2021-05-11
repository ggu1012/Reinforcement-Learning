import tkinter as tk
from tkinter import Button
import time
import numpy as np
from PIL import ImageTk, Image
PhotoImage = ImageTk.PhotoImage

UNIT = 100  # Number of pixels for each grid


class GraphicDisplay(tk.Tk):
    def __init__(self, Env, agent):
        super(GraphicDisplay, self).__init__()
        self.env = Env
        self.HEIGHT, self.WIDTH = self.env.size()
        self.title('Policy Iteration')
        self.geometry('{0}x{1}'.format(self.WIDTH * UNIT, self.HEIGHT * UNIT + 50))
        self.texts = []
        self.arrows = []

        self.agent = agent
        self.is_moving = 0
        (self.up, self.down, self.left, self.right), self.shapes = self.load_images()
        self.canvas = self._build_canvas()


    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height= self.HEIGHT * UNIT+50,
                           width= self.WIDTH * UNIT)
        policy_button = Button(self, text="move", command=self.move_by_policy)
        policy_button.configure(width=10, activebackground="#33B5E5")
        canvas.create_window(self.WIDTH * UNIT * 0.62, self.HEIGHT * UNIT + 30,
                                window=policy_button)


        # 그리드 생성
        for col in range(0, self.WIDTH * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = col, 0, col, self.HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for row in range(0, self.HEIGHT * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, row, self.WIDTH * UNIT, row
            canvas.create_line(x0, y0, x1, y1)

        self.goal = []
        for k in range(len(self.env.goal)):
            i,j = self.env.goal[k]
            self.goal.append(canvas.create_image(self.matrix2image_index(i,j), image=self.shapes[3]))


        self.obstacles = []
        for k in range(len(self.env.obstacles)):
            i,j = self.env.obstacles[k]
            self.obstacles.append(canvas.create_image(self.matrix2image_index(i,j), image=self.shapes[1]))
        canvas.pack()
        return canvas

    def matrix2image_index(self,i,j):
        return (j*UNIT+UNIT/2, i*UNIT+UNIT/2)


    def load_images(self):
        up = PhotoImage(Image.open("./img/up.png").resize((13, 13)))
        right = PhotoImage(Image.open("./img/right.png").resize((13, 13)))
        left = PhotoImage(Image.open("./img/left.png").resize((13, 13)))
        down = PhotoImage(Image.open("./img/down.png").resize((13, 13)))
        rectangle = PhotoImage(Image.open("./img/rectangle.png").resize((65, 65)))
        triangle = PhotoImage(Image.open("./img/triangle.png").resize((65, 65)))
        circle = PhotoImage(Image.open("./img/circle.png").resize((65, 65)))
        goal_grid = PhotoImage(Image.open("./img/goal.png").resize((100, 100)))
        return (up, down, left, right), (rectangle, triangle, circle, goal_grid)


    def text_value(self, row, col, contents, font='Helvetica', size=10,
                   style='normal', anchor="nw"):
        origin_x, origin_y = 40, 40       #85, 70
        x, y = origin_x + (UNIT * col), origin_y + (UNIT * row)
        font = (font, str(size), style)
        text = self.canvas.create_text(x, y, fill="black", text=format(contents, '.2f'),
                                       font=font, anchor=anchor)
        return self.texts.append(text)


    def print_value_table(self):
        for i in range(self.HEIGHT):
            for j in range(self.WIDTH):
                self.text_value(i, j, self.agent.V_values[i][j])



    def rectangle_move(self, state, motion):
        # base_action = np.array([0, 0])
        next_state, r = self.env.interaction(state,motion)
        movement = (np.array(next_state)-np.array(state)).tolist()
        self.render()
        self.canvas.move(self.rectangle, movement[1]*UNIT, movement[0]*UNIT)


    def find_rectangle(self):
        temp = self.canvas.coords(self.rectangle)
        x = (temp[0] / 100) - 0.5
        y = (temp[1] / 100) - 0.5
        return int(y), int(x)


    def move_by_policy(self):
        print("move_by_policy")
        self.is_moving = 1

        [start_row, start_col] = self.agent.initialize_episode()
        self.rectangle = self.canvas.create_image(self.matrix2image_index(start_row, start_col), image=self.shapes[0])

        x, y = self.canvas.coords(self.rectangle)
        print(x,y)
        i,j = start_row, start_col
        while [i,j] not in self.env.goal:
            self.after(100,self.rectangle_move(self.agent.state,
                        self.agent.ACTIONS[self.agent.get_action([i, j])]))
            i, j = self.agent.state = self.find_rectangle()
        self.is_moving = 0

    def render(self):
        time.sleep(0.35)
        self.canvas.tag_raise(self.rectangle)
        self.update()


