import numpy as np
from visdom import Visdom
import pdb
class Plot(object):
    def __init__(self, title, port=8080):
        self.viz = Visdom(port=port)
        self.windows = {}
        self.title = title


    def update_scatterplot(self, name, x, y):
        self.viz.line(
        X=np.asarray([x]),
        Y=np.asarray([y]),
        win=name,
        update='append'
        )

