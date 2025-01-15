import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
import matplotlib.pyplot as plt

def select_grid_element(pos_obj, pos_area, pos_tcp):
    # Berechnung der relativen Position und des Zielvektors
    rel_pos = np.array(pos_area) - np.array(pos_obj)
    vec_obj_to_tcp = np.array(pos_tcp) - np.array(pos_obj)

    print(f"Relative Position: {rel_pos}")
    print(f"TCP to Object Vector: {vec_obj_to_tcp}")
    
    # Winkelberechnung zwischen rel_pos und vec_obj_to_tcp
    theta = np.arctan2(rel_pos[1], rel_pos[0]) - np.arctan2(vec_obj_to_tcp[1], vec_obj_to_tcp[0])
    
    # Umwandlung des Winkels in den Bereich 0 bis 2pi
    theta = np.mod(theta, 2 * np.pi)

    # Referenzwinkel für die Grid-Elemente
    ref_pos_x = np.array([pos_obj[0], pos_obj[1] + 1]) - np.array(pos_obj)
    theta_ref = np.arctan2(rel_pos[1], rel_pos[0]) - np.arctan2(ref_pos_x[1], ref_pos_x[0])
    
    # Umwandlung des Winkels in den Bereich 0 bis 2pi
    theta_ref = np.mod(theta_ref, 2 * np.pi)

    print(f"Reference Angle: {theta_ref * 180.0 / np.pi}° or {theta_ref} rad for ref pos {ref_pos_x}")
    print(f"Angle: {theta * 180.0 / np.pi}° or {theta} rad")

    # Berechnen des Startwinkels
    theta_start = np.pi - theta_ref
    print(f"Start Angle: {theta_start * 180.0 / np.pi}° or {theta_start} rad")
    
    # Bestimmung des Grid-Elements basierend auf dem Winkel
    if -np.pi/6 <= theta < np.pi/6:
        return 5, theta  # Mitte-Ost
    elif np.pi/6 <= theta < np.pi/3:
        return 3, theta  # Nord-Ost
    elif np.pi/3 <= theta < 2*np.pi/3:
        return 2, theta  # Nord 
    elif 2*np.pi/3 <= theta <= np.pi:
        return 1, theta  # Nord-West
    elif -np.pi/3 <= theta < -np.pi/6:
        return 7, theta  # Süd-Ost
    elif -2*np.pi/3 <= theta < -np.pi/3:
        return 8, theta  # Süd
    elif -np.pi <= theta < -2*np.pi/3:
        return 9, theta  # Süd-West
    else:
        return 6, theta  # West

def main():
   # Define positions
    obj_pos = (5, 5)  # Object is at the center
    area_pos = (5, 3)  # Area is top-right
    tcp_pos = (5,3)  # TCP is bottom-left

    # create ref pos
    ref_pos = (obj_pos[0], obj_pos[1] + 1)

    # Calculate grid element
    grid_element, theta = select_grid_element(obj_pos, area_pos, tcp_pos)
    print(f"Selected Grid Element: {grid_element} with angle {theta * 180.0 / np.pi}° or {theta} rad")

    # Plotting
    fig, ax = plt.subplots()
    # Drawing the environment
    for i in range(3):
        for j in range(3):
            ax.add_patch(plt.Rectangle((obj_pos[0] - 1.5 + i, obj_pos[1] - 1.5 + j), 1, 1, edgecolor='gray', facecolor='none'))

    # Highlight the selected grid element
    # Berechnung der richtigen Zeilen- und Spaltenindizes für das ausgewählte Gridelement
    row = (grid_element - 1) // 3  # Zeilenindex von 0 bis 2
    col = (grid_element - 1) % 3   # Spaltenindex von 0 bis 2
    ax.add_patch(plt.Rectangle((obj_pos[0] - 1.5 + col, obj_pos[1] - 1.5 + row), 1, 1, edgecolor='red', facecolor='lightblue', label='Selected Grid'))

    # Plot positions
    ax.plot(*obj_pos, 'ro', label='Object')
    ax.plot(*area_pos, 'go', label='Area')
    ax.plot(*tcp_pos, 'bo', label='TCP')
    ax.plot(*ref_pos, 'ko', label='Reference Position')

    # Set limits and grid
    ax.set_xlim(-5, 15)
    ax.set_ylim(-5, 15)
    ax.set_xticks(np.arange(-5, 16, 1))
    ax.set_yticks(np.arange(-5, 16, 1))
    ax.grid(True)

    # Labels and legends
    ax.set_title(f'Selected Grid Element: {grid_element}')
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
