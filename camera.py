import jax



def create_camera_directions(camera_resolution_width, camera_resolution_height, camera_focal_x, camera_focal_y):
    u = jax.device_put(jax.numpy.arange(camera_resolution_width, dtype=jax.numpy.float32))
    v = jax.device_put(jax.numpy.arange(camera_resolution_height, dtype=jax.numpy.float32))

    camera_center_x = (camera_resolution_width - 1) / 2.0
    camera_center_y = (camera_resolution_height - 1) / 2.0

    x = ((u - camera_center_x) / camera_focal_x)
    y = ((v - camera_center_y) / camera_focal_y)
    z = jax.device_put(jax.numpy.ones((camera_resolution_height, camera_resolution_width), dtype=jax.numpy.float32))

    xx, yy = jax.device_put(jax.numpy.meshgrid(x, y, indexing="xy"))

    camera_directions = jax.device_put(jax.numpy.stack((xx, yy, z), axis=2))
    camera_directions /= jax.numpy.linalg.norm(camera_directions, axis=2)[:, :, jax.numpy.newaxis]

    return camera_directions

def create_rotation_matrix(direction, up_direction):
    if abs(jax.numpy.dot(direction, up_direction)) > 0.999:
        up_direction = jax.numpy.array([1.0, 0.0, 0.0]) if abs(direction[0]) < 0.9 else jax.numpy.array([0.0, 1.0, 0.0])

    right_direction = jax.numpy.cross(direction, up_direction)
    right_direction /= jax.numpy.linalg.norm(right_direction)

    return jax.numpy.stack([right_direction, jax.numpy.cross(right_direction, direction), direction], axis=1)

def create_camera_rotation_matrix(original_direction, target_direction, up_direction):
    original_direction /= jax.numpy.linalg.norm(original_direction)
    target_direction /= jax.numpy.linalg.norm(target_direction)
    up_direction /= jax.numpy.linalg.norm(up_direction)

    return (create_rotation_matrix(target_direction, up_direction) @ create_rotation_matrix(original_direction, up_direction).T)

