import numpy



def create_camera_directions(camera_resolution_width, camera_resolution_height, camera_focal_x, camera_focal_y):
    u = numpy.arange(camera_resolution_width, dtype=numpy.float32)
    v = numpy.arange(camera_resolution_height, dtype=numpy.float32)

    camera_center_x = (camera_resolution_width - 1) / 2.0
    camera_center_y = (camera_resolution_height - 1) / 2.0

    x = ((u - camera_center_x) / camera_focal_x)
    y = ((v - camera_center_y) / camera_focal_y)
    z = numpy.ones((camera_resolution_height, camera_resolution_width), dtype=numpy.float32)

    xx, yy = numpy.meshgrid(x, y, indexing="xy")

    camera_directions = numpy.stack((xx, yy, z), axis=2)
    camera_directions /= numpy.linalg.norm(camera_directions, axis=2)[:, :, numpy.newaxis]

    return camera_directions

def create_rotation_matrix(direction, up_direction=numpy.array([0.0, 0.0, 1.0])):
    if abs(numpy.dot(direction, up_direction)) > 0.999:
        up_direction = numpy.array([1.0, 0.0, 0.0]) if abs(direction[0]) < 0.9 else numpy.array([0.0, 1.0, 0.0])

    right_direction = numpy.cross(direction, up_direction)
    right_direction /= numpy.linalg.norm(right_direction)

    return numpy.stack([right_direction, numpy.cross(right_direction, direction), direction], axis=1)

def create_camera_rotation_matrix(original_direction, target_direction, up_direction):
    original_direction /= numpy.linalg.norm(original_direction)
    target_direction /= numpy.linalg.norm(target_direction)
    up_direction /= numpy.linalg.norm(up_direction)

    return (create_rotation_matrix(target_direction, up_direction) @ create_rotation_matrix(original_direction, up_direction).T)
