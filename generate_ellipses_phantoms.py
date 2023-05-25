import tomopy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

SAVE_AS_PNG = FALSE

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

if __name__ == '__main__':
    # array_to_params = tomopy.misc.phantom._array_to_params()
    # phantom = tomopy.misc.phantom.shepp2d()
    # phantom = tomopy.shepp2d()
    # phantom3d = tomopy.shepp3d()

    array_size = 256

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

    num_ellipses = 10

    num_cubes = 512

    for cube_number in range(num_cubes):
        ellipse_list = []

        # Big ellipses
        for ellipse_num in range(num_ellipses):
            A = np.random.uniform(low=-1, high=1)
            a = np.random.uniform(low=0, high=1)
            b = np.random.uniform(low=0, high=1)
            c = np.random.uniform(low=0, high=1)
            x0 = np.random.uniform(low=-0.5, high=0.5)
            y0 = np.random.uniform(low=-0.5, high=0.5)
            z0 = np.random.uniform(low=-0.5, high=0.5)
            phi = np.random.uniform(low=0,high=180)
            theta = np.random.uniform(low=0,high=180)
            psi = np.random.uniform(low=0,high=180)

            ellipse_row = [A, a, b, c, x0, y0, z0, phi, theta, psi]

            ellipse_list.append(ellipse_row)

        # Small ellipses
        for ellipse_num in range(num_ellipses * 10):
            A = np.random.uniform(low=-0.5, high=0.5)
            a = np.random.uniform(low=0, high=0.2)
            b = np.random.uniform(low=0, high=0.2)
            c = np.random.uniform(low=0, high=0.2)
            x0 = np.random.uniform(low=-0.5, high=0.5)
            y0 = np.random.uniform(low=-0.5, high=0.5)
            z0 = np.random.uniform(low=-0.5, high=0.5)
            phi = np.random.uniform(low=0,high=180)
            theta = np.random.uniform(low=0,high=180)
            psi = np.random.uniform(low=0,high=180)

            ellipse_row = [A, a, b, c, x0, y0, z0, phi, theta, psi]

            ellipse_list.append(ellipse_row)

        ellipse_array = np.array(ellipse_list)

        params = tomopy.misc.phantom._array_to_params(ellipse_array)

        if 1 == 1:
            z_array_size = array_size
        else:
            z_array_size = 5

        # half_point = int(float(array_size) / 2.)
        # half_point = int((float(z_array_size) / 2.) - 0.1)

        # ellipse_cube = tomopy.misc.phantom.phantom((array_size,array_size,z_array_size), params)[:,:,half_point]
        ellipse_cube = tomopy.misc.phantom.phantom((array_size, array_size, z_array_size), params)

        max_cube = np.max(ellipse_cube)
        min_cube = np.min(ellipse_cube)

        rescaled_cube = (((ellipse_cube - min_cube) / (max_cube - min_cube))*255.).astype(np.uint8)



        if SAVE_AS_PNG:
            base_dir = 'trainset256/'
            cube_filename = base_dir + str(cube_number) + '.png'

            # np.save(cube_filename, rescaled_cube)

            # combined_file_name = './Combined images/Output/Output ' + str(index) + '.png'
            combined_image = Image.fromarray(rescaled_cube)
            combined_image.save(cube_filename, "PNG")
        else:
            base_dir = 'cubetraindata/'
            cube_filename = base_dir + str(cube_number) + '.npy'
            np.save(cube_filename, rescaled_cube)

print('Done')
