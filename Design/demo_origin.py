class point():
    def __init__(self, x, y):
        self.x = x
        self.y = y


class line():
    def __init__(self, start, end):
        self.start = start
        self.end = end


class rec():
    def __init__(self, leftup, width, height):
        self.leftup = leftup
        self.width = width
        self.height = height


class draw():
    def __init__(self):
        self.line_list = []
        self.rec_list = []
        self.draw_type = input()

        self.mouse_down()
        self.mouse_up()
        self.onpaint()

    def mouse_down(self):
        input_1, input_2 = input().split()
        self.key1 = point(input_1, input_2)

    def mouse_up(self):
        input_1, input_2 = input().split()
        self.key2 = point(input_1, input_2)
        if self.draw_type == 'line':
            self.line_list.append(line(self.key1, self.key2))
        elif self.draw_type == 'rec':
            self.rec_list.append(rec(self.key1, self.key2.x-self.key1.x, self.key2.y-self.key1.y))
        else:
            pass

    def onpaint(self):
        for line in self.line_list:
            print('draw line', line.start.x, line.start.y, line.end.x, line.end.y)
        for rec in self.rec_list:
            print('draw rec', rec.leftup.x, rec.leftup.y, rec.width, rec.height)


draw()
