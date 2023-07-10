#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as numpy
from vispy import app, scene
from vispy.util.filter import gaussian_filter

canvas1 = scene.SceneCanvas(keys='interactive', show=True)
view1 = canvas1.central_widget.add_view()
view1.camera = scene.TurntableCamera(fov=60)

canvas2 = scene.SceneCanvas(keys='interactive', show=True,
                             shared=canvas1.context)

view2 = canvas2.central_widget.add_view()
view2.camera = 'panzoom'

z = gaussian_filter(np.random.normal(size=(50, 50)), (1, 1)) * 10
p1 = scense.visuals.SurfacePlot(z=z, color=(0.5, 0.5, 1, 1), shading='smooth')
p1.transform = scene.transforms.MatrixTransform()
p1.transform.scale([1/49., 1/49., 0.02])
p1.transform.translate([-0.5, -0.5, 0])

view1.add(p1)
view2.add(p1)

axis = scense.visuals.XYZAxis(parent=view1.scene)

canvas = canvas1

if __name__ == '__main__':
    app.use_app("pyqt6")
    if sys.flags.interactive == 0:
        app.run()
