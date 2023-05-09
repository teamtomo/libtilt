from libtilt.shapes.shapes_2d import circle, square, rectangle, wedge


def _circle():
    result = circle(
        radius=5,
        image_shape=10,
        center=None,
        smoothing_radius=5,
    )
    import napari
    viewer = napari.Viewer()
    viewer.add_image(result.numpy())
    napari.run()


if __name__ == '__main__':
    _circle()
