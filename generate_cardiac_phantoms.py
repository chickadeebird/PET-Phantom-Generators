import tomopy
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
from scipy.ndimage import median_filter
import math

def plot_two(img1, img2, figsize=(10, 6), titles=['', ''],
             cmap=plt.cm.gray):
    ax1 = plt.figure(figsize=figsize)
    ax1 = plt.subplot(121)
    ax1.imshow(img1, cmap=cmap)
    ax1.set_title(titles[0])
    ax2 = plt.subplot(122)
    ax2.imshow(img2, cmap=cmap)
    ax2.set_title(titles[1])
    plt.show()

def rotate_cube_old(input_matrix, deg_angle, axis):
    d = len(input_matrix)
    h = len(input_matrix[0])
    w = len(input_matrix[0][0])
    min_new_x = 0
    max_new_x = 0
    min_new_y = 0
    max_new_y = 0
    min_new_z = 0
    max_new_z = 0
    new_coords = []
    angle = math.radians(deg_angle)

    for z in range(d):
        for y in range(h):
            for x in range(w):

                new_x = None
                new_y = None
                new_z = None

                if axis == "x":
                    new_x = int(round(x))
                    new_y = int(round(y * math.cos(angle) - z * math.sin(angle)))
                    new_z = int(round(y * math.sin(angle) + z * math.cos(angle)))
                elif axis == "y":
                    new_x = int(round(z * math.sin(angle) + x * math.cos(angle)))
                    new_y = int(round(y))
                    new_z = int(round(z * math.cos(angle) - x * math.sin(angle)))
                elif axis == "z":
                    new_x = int(round(x * math.cos(angle) - y * math.sin(angle)))
                    new_y = int(round(x * math.sin(angle) + y * math.cos(angle)))
                    new_z = int(round(z))

                val = input_matrix.item((z, y, x))
                new_coords.append((val, new_x, new_y, new_z))
                if new_x < min_new_x: min_new_x = new_x
                if new_x > max_new_x: max_new_x = new_x
                if new_y < min_new_y: min_new_y = new_y
                if new_y > max_new_y: max_new_y = new_y
                if new_z < min_new_z: min_new_z = new_z
                if new_z > max_new_z: max_new_z = new_z

    new_x_offset = abs(min_new_x)
    new_y_offset = abs(min_new_y)
    new_z_offset = abs(min_new_z)

    new_width = abs(min_new_x - max_new_x)
    new_height = abs(min_new_y - max_new_y)
    new_depth = abs(min_new_z - max_new_z)

    rotated = np.empty((new_depth + 1, new_height + 1, new_width + 1))
    rotated.fill(0)
    for coord in new_coords:
        val = coord[0]
        x = coord[1]
        y = coord[2]
        z = coord[3]

        if rotated[new_z_offset + z][new_y_offset + y][new_x_offset + x] == 0:
            rotated[new_z_offset + z][new_y_offset + y][new_x_offset + x] = val

    return rotated

def rotate_cube(data, angle, axes):
    """
    Rotate a `data` based on rotating coordinates.
    """

    # Create grid of indices
    shape = data.shape
    d1, d2, d3 = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]

    # Rotate the indices
    if 1 == 0:
        d1r = rotate(d1, angle=angle, axes=axes, reshape=False)
        d2r = rotate(d2, angle=angle, axes=axes, reshape=False)
        d3r = rotate(d3, angle=angle, axes=axes, reshape=False)

        return data[d1r, d2r, d3r]
    else:
        rot_data = rotate(data,angle=angle, axes=axes, reshape=False)
        rot_data = median_filter(rot_data,size=3)

    # Round to integer indices
    # d1r = np.round(d1r)
    # d2r = np.round(d2r)
    # d3r = np.round(d3r)

    # d1r = np.clip(d1r, 0, shape[0])
    # d2r = np.clip(d2r, 0, shape[1])
    # d3r = np.clip(d3r, 0, shape[2])

    return rot_data

if __name__ == '__main__':
    # array_to_params = tomopy.misc.phantom._array_to_params()
    # phantom = tomopy.misc.phantom.shepp2d()
    # phantom = tomopy.shepp2d()
    # phantom3d = tomopy.shepp3d()

    array_size = 128

    # empty_box = np.zeros((array_size,array_size,array_size),dtype=np.float32)

    shepp_array = [
        [1., .6900, .920, .810, 0., 0., 0., 90., 90., 90.],
        [-.8, .6624, .874, .780, 0., -.0184, 0., 90., 90., 90.],
        [-.2, .1100, .310, .220, .22, 0., 0., -108., 90., 100.],
        [-.2, .1600, .410, .280, -.22, 0., 0., 108., 90., 100.],
        [.1, .2100, .250, .410, 0., .35, -.15, 90., 90., 90.],
        [.1, .0460, .046, .050, 0., .1, .25, 90., 90., 90.],
        [.1, .0460, .046, .050, 0., -.1, .25, 90., 90., 90.],
        [.1, .0460, .023, .050, -.08, -.605, 0., 90., 90., 90.],
        [.1, .0230, .023, .020, 0., -.606, 0., 90., 90., 90.],
        [.1, .0230, .046, .020, .06, -.605, 0., 90., 90., 90.]]

    num_ellipses = 1

    num_cubes = 256

    for cube_number in range(num_cubes):
        print('Cube number : ' + str(cube_number))
        ellipse_list = []
        chamber_list = []

        # Big ellipses

        A = np.random.uniform(low=0.5, high=1)
        original_wall_activity = A
        a = np.random.uniform(low=0.35, high=0.55)
        b = np.random.uniform(low=0.35, high=0.55)
        c = np.random.uniform(low=0.5, high=0.7)
        x0 = np.random.uniform(low=0.1, high=0.2)
        y0 = np.random.uniform(low=0.1, high=0.2)
        z0 = np.random.uniform(low=-0.1, high=0.1)
        x0 = 0
        y0 = 0
        z0 = 0
        phi = np.random.uniform(low=30,high=60)
        theta = np.random.uniform(low=30,high=60)
        psi = np.random.uniform(low=30,high=60)
        phi = 0
        theta = 0
        psi = 0

        ellipse_row1 = [A, a, b, c, x0, y0, z0, phi, theta, psi]

        ellipse_list.append(ellipse_row1)

        cardiac_wall_thickness = np.random.uniform(low=0.05, high=0.15)

        ellipse_row2 = [-A, a-cardiac_wall_thickness, b-cardiac_wall_thickness, c-cardiac_wall_thickness, x0, y0, z0, phi, theta, psi]
        ellipse_row_chambers = [1., a - cardiac_wall_thickness, b - cardiac_wall_thickness, c - cardiac_wall_thickness, x0, y0,
                        z0, phi, theta, psi]

        ellipse_list.append(ellipse_row2)
        chamber_list.append(ellipse_row_chambers)
        # chamber_list.append(ellipse_row_chambers)

        ellipse_array = np.array(ellipse_list)
        chamber_array = np.array(chamber_list)

        params = tomopy.misc.phantom._array_to_params(ellipse_array)
        chamber_params = tomopy.misc.phantom._array_to_params(chamber_array)

        if 1 == 0:
            z_array_size = array_size
        else:
            z_array_size = 128

        # half_point = int(float(array_size) / 2.)
        # half_point = int((float(z_array_size) / 2.) - 0.1)

        ellipse_cube = tomopy.misc.phantom.phantom((array_size, array_size, z_array_size), params)
        chamber_cube = tomopy.misc.phantom.phantom((array_size, array_size, z_array_size), chamber_params)

        orthogonal_septal_matrix = np.zeros((array_size, array_size, z_array_size),
                                            dtype=ellipse_cube.dtype)
        half_z = int(z_array_size / 2)
        half_x = int(array_size / 2)
        z_septal_width = 2
        x_septal_width = 1

        orthogonal_septal_matrix[:, :, half_z - z_septal_width:half_z + z_septal_width] = original_wall_activity
        orthogonal_septal_matrix[half_x - x_septal_width:half_x + x_septal_width, :, :] = original_wall_activity

        slice_number = 70
        half_orth = orthogonal_septal_matrix[slice_number,:,:].copy()
        half_chamber = chamber_cube[slice_number,:,:].copy()
        half_ellipse = ellipse_cube[slice_number, :, :].copy()

        orthogonal_septal_matrix = orthogonal_septal_matrix * chamber_cube
        half_multiply = orthogonal_septal_matrix[slice_number, :, :].copy()

        ellipse_and_septum_cube = np.maximum(ellipse_cube, orthogonal_septal_matrix)

        half_ellipse_and_septum = ellipse_and_septum_cube[slice_number, :, :].copy()

        max_number_of_lesions = 10
        min_number_of_lesions = 4
        number_of_lesions = np.random.randint(low=min_number_of_lesions, high=max_number_of_lesions)
        lesion_list = []
        print('Number of lesions : ' + str(number_of_lesions))

        for lesion_number in range(number_of_lesions):
            positive_coords = np.argwhere(ellipse_and_septum_cube > 0.01)

            random_coord_loc = np.random.randint(low=0, high=len(positive_coords))

            random_coord = positive_coords[random_coord_loc]

            A = np.random.uniform(low=0.1, high=original_wall_activity)
            a = np.random.uniform(low=0.05, high=0.1)
            b = np.random.uniform(low=0.05, high=0.1)
            c = np.random.uniform(low=0.05, high=0.1)
            # x0 = (random_coord[0] - half_x) / array_size
            # y0 = (random_coord[1] - half_x) / array_size
            # z0 = (random_coord[2] - half_z) / z_array_size
            x0 = np.random.uniform(low=-0.5, high=0.5)
            y0 = np.random.uniform(low=-0.5, high=0.5)
            z0 = np.random.uniform(low=-0.7, high=0.7)
            phi = np.random.uniform(low=0, high=180)
            theta = np.random.uniform(low=0, high=180)
            psi = np.random.uniform(low=0, high=180)

            lesion_row = [A, a, b, c, x0, y0, z0, phi, theta, psi]

            lesion_list.append(lesion_row)

        lesion_array = np.array(lesion_list)

        lesion_params = tomopy.misc.phantom._array_to_params(lesion_array)

        lesion_cube = tomopy.misc.phantom.phantom((array_size, array_size, z_array_size), lesion_params)

        ellipse_and_septum_cube_with_lesions = ellipse_and_septum_cube - lesion_cube

        ellipse_and_septum_cube_with_lesions[ellipse_and_septum_cube_with_lesions < 0] = 0

        if 1 == 0:
            deg_angle = 30
            axis = "x"
            ellipse_and_septum_cube_with_lesions = rotate_cube(ellipse_and_septum_cube_with_lesions, deg_angle, axis)
            deg_angle = 20
            axis = "z"
            ellipse_and_septum_cube_with_lesions = rotate_cube(ellipse_and_septum_cube_with_lesions, deg_angle, axis)

            # ellipse_cube_with_lesion = np.maximum(ellipse_cube_with_lesion,orthogonal_septal_matrix) * 255.

            # mid_slice = ellipse_cube_with_lesion[half_x,:,:]

        phi = np.random.uniform(low=-30, high=30)
        theta = np.random.uniform(low=-30, high=30)
        psi = np.random.uniform(low=-30, high=30)
        # angle = 30
        axes = (0,1)
        ellipse_and_septum_cube_with_lesions = rotate_cube(ellipse_and_septum_cube_with_lesions, phi, axes)
        # angle = 30
        axes = (0, 2)
        ellipse_and_septum_cube_with_lesions = rotate_cube(ellipse_and_septum_cube_with_lesions, theta, axes)
        # angle = 30
        axes = (1, 2)
        ellipse_and_septum_cube_with_lesions = rotate_cube(ellipse_and_septum_cube_with_lesions, psi, axes)

        x_trans = np.random.randint(low=-20, high=20)
        y_trans = np.random.randint(low=-20, high=20)
        z_trans = np.random.randint(low=-20, high=20)

        ellipse_and_septum_cube_with_lesions = np.roll(ellipse_and_septum_cube_with_lesions, x_trans, axis=1)
        ellipse_and_septum_cube_with_lesions = np.roll(ellipse_and_septum_cube_with_lesions, y_trans, axis=0)
        ellipse_and_septum_cube_with_lesions = np.roll(ellipse_and_septum_cube_with_lesions, z_trans, axis=2)

        # Liver

        A = np.random.uniform(low=0.3, high=0.5)
        a = np.random.uniform(low=0.3, high=0.4)
        b = np.random.uniform(low=0.3, high=0.4)
        c = np.random.uniform(low=0.3, high=0.4)
        x0 = np.random.uniform(low=0., high=0.25)
        y0 = np.random.uniform(low=0., high=0.25)
        z0 = 1.
        phi = np.random.uniform(low=30, high=60)
        theta = np.random.uniform(low=30, high=60)
        psi = np.random.uniform(low=30, high=60)
        # phi = 0
        # theta = 0
        # psi = 0

        liver_list = []
        ellipse_row_liver = [A, a, b, c, x0, y0, z0, phi, theta, psi]

        liver_list.append(ellipse_row_liver)
        liver_array = np.array(liver_list)
        liver_params = tomopy.misc.phantom._array_to_params(liver_array)

        liver_cube = tomopy.misc.phantom.phantom((array_size, array_size, z_array_size), liver_params)

        full_cube = np.maximum(ellipse_and_septum_cube_with_lesions, liver_cube)

        if 1 == 0:
            x_size, y_size, z_size = ellipse_and_septum_cube_with_lesions.shape
            mid_x = int(x_size / 2)
            mid_y = int(y_size / 2)
            mid_z = int(z_size / 2)
            ellipse_and_septum_cube_with_lesions = ellipse_and_septum_cube_with_lesions[mid_x-half_x:mid_x+half_x, mid_y-half_x:mid_y+half_x, mid_z-half_z:mid_z+half_z]

        # max_cube = np.max(ellipse_cube)
        # min_cube = np.min(ellipse_cube)

        # rescaled_cube = (((ellipse_cube - min_cube) / (max_cube - min_cube))*255.).astype(np.uint8)

        base_dir = 'cardiacPET128/'

        cube_filename = base_dir + str(cube_number) + '.npy'
        # lesion_filename = base_dir + str(cube_number) + 'l.npy'

        full_cube[full_cube < 0] = 0
        full_cube[full_cube > original_wall_activity] = original_wall_activity

        np.save(cube_filename, full_cube)
        # np.save(lesion_filename, lesion_cube * 255.)

        a=1

        # combined_file_name = './Combined images/Output/Output ' + str(index) + '.png'
        # combined_image = Image.fromarray(rescaled_cube)
        # combined_image.save(cube_filename, "PNG")

print('Done')