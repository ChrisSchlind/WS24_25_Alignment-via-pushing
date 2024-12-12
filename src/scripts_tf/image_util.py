import cv2


def draw_coordinate_frame(image, pose, workspace_bounds, resolution):
    """Draw coordinate frame on image with x-axis (red) and y-axis (green).
    
    Args:
        image: RGB image to draw on
        pose: 4x4 transformation matrix
        workspace_bounds: List of [min, max] pairs for x, y, z dimensions
        resolution: [height, width] of the image
    """
    # Get position and rotation
    position = pose[:3, 3]
    rotation = pose[:3, :3]
    
    # Project position to pixel coordinates
    x_bounds, y_bounds, _ = workspace_bounds
    height, width = resolution
    
    x_scale = width / (x_bounds[1] - x_bounds[0])
    y_scale = height / (y_bounds[1] - y_bounds[0])
    
    px = int((position[0] - x_bounds[0]) * x_scale)
    py = int((position[1] - y_bounds[0]) * y_scale)
    
    # Define axis lengths (in pixels)
    axis_length = 20
    
    # Draw axes (using rotation matrix)
    # X-axis in red
    x_end = px + int(rotation[0, 0] * axis_length)
    y_end = py + int(rotation[1, 0] * axis_length)
    cv2.line(image, (px, py), (x_end, y_end), (0, 0, 255), 2)
    
    # Y-axis in green
    x_end = px + int(rotation[0, 1] * axis_length)
    y_end = py + int(rotation[1, 1] * axis_length)
    cv2.line(image, (px, py), (x_end, y_end), (0, 255, 0), 2)
    
    # Draw origin point
    cv2.circle(image, (px, py), 3, (255, 255, 255), -1)
    
    return image
