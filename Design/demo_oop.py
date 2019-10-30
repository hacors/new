class shape():
    def draw():
        raise NotImplementedError


class point():
    def __init__(self, x, y):
        self.x = x
        self.y = y


class line(shape):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def draw(self):
        print('draw line', self.start.x, self.start.y, self.end.x, self.end.y)


class rec(shape):
    def __init__(self, leftup, width, height):
        self.leftup = leftup
        self.width = width
        self.height = height

    def draw(self):
        print('draw rec', self.leftup.x, self.leftup.y, self.width, self.height)


class mydraw():
    def __init__(self):
        self.shape_list = []
        self.draw_type = input()

        self.mouse_down()
        self.mouse_up()
        self.onpaint()

    def mouse_down(self):
        input_1, input_2 = list(map(int, input().split()))
        self.key1 = point(input_1, input_2)

    def mouse_up(self):
        input_1, input_2 = list(map(int, input().split()))
        self.key2 = point(input_1, input_2)
        if self.draw_type == 'line':
            self.shape_list.append(line(self.key1, self.key2))
        elif self.draw_type == 'rec':
            self.shape_list.append(rec(self.key1, self.key2.x-self.key1.x, self.key2.y-self.key1.y))
        else:
            pass

    def onpaint(self):
        for shape in self.shape_list:
            shape.draw()


mydraw()
