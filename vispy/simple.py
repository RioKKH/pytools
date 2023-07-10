import math
from vispy import app, gloo


class Canvas(app.Canvas):

    def __init__(self, *args, **kwargs):
        app.Canvas.__init__(self, *args, **kwargs)
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.tick = 0

    def on_draw(self, event):
        gloo.clear(color=True)

    def on_timer(self, event):
        self.tick += 1 / 60.0
        c = abs(math.sin(self.tick))
        gloo.set_clear_color((c, c, c, 1))
        self.update()

if __name__ == '__main__':
    # https://github.com/vispy/vispy/issues/2068
    app.use_app("pyqt6")
    #app.use_app("pyglet")
    #canvas = Canvas(keys='interactive')
    canvas = Canvas(keys='interactive', always_on_top=True)
    canvas.show()
    app.run()
