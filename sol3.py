from scipy import ndimage, signal
import numpy as np
from scipy.misc import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import os


rgb2yiqMatrix = [[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]]
Matrix = np.array(rgb2yiqMatrix)
GREY_LEVEL_MAX_VAL = 256
LOWEST_SIZE = 16

def read_image(filename, representation):
    """
    function which reads an image file and converts it into a given representation.
    This function returns an image, normalized to the range [0, 1].
    """
    im = imread(filename).astype(np.float64) / (GREY_LEVEL_MAX_VAL - 1)
    if (representation == 1):
        im_g = rgb2gray(im)
        return im_g
    return im


def create_filter(filter_size):
    """
    :param filter_size
    :return: the filter.
    """
    filter = [1 / 2, 1 / 2]
    conv_with = [1 / 2, 1 / 2]
    for _ in range(filter_size - 2):
        filter = signal.convolve(filter, conv_with)
    return filter.reshape(1, len(filter))


def reduce(im, max_levels, filter_vec):
    """
    finds the pyr.
    :return: pyr
    """
    pyr = [im]
    cur_im = im
    for _ in range(max_levels - 1):
        conv_vec = ndimage.filters.convolve(filter_vec, filter_vec.T)
        cur_im = ndimage.filters.convolve(cur_im, conv_vec)
        sampled_pic = cur_im[0: cur_im.shape[0]: 2]
        sampled_pic = sampled_pic.T[0: cur_im.shape[1]: 2]
        x_size, y_size = sampled_pic.shape
        if x_size < LOWEST_SIZE or y_size < LOWEST_SIZE:
            break
        cur_im = sampled_pic.T
        pyr.append(cur_im)
    return pyr


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    construct a Gaussian pyramid pyramid of a given image.
    :param im: a grayscale image with double values in [0, 1]
    (e.g. the output of ex1’s read_image with the representation set to 1).
    :param max_levels: the maximal number of levels1 in the resulting pyramid
    :param filter_size: the size of the Gaussian filter (an odd scalar that
    represents a squared filter) to be used in constructing the pyramid filter.
    :return: pyr, filter_vec
    """
    filter_vec = create_filter(filter_size)
    pyr = reduce(im, max_levels, filter_vec)
    return pyr, filter_vec


def expand_single_pic(image, filter_vec, x_flag, y_flag):
    x_size, y_size = image.shape
    new_img = np.zeros((2 * x_size - x_flag, 2 * y_size - y_flag))
    for row in range(x_size):
        for pixel in range(y_size):
            new_img[2 * row][2 * pixel] = image[row][pixel]
    conv_vec = ndimage.filters.convolve(new_img, filter_vec)
    cur_im = ndimage.filters.convolve(conv_vec, filter_vec.T)
    x_flag = x_size % 2
    y_flag = y_size % 2
    return cur_im, x_flag, y_flag

def expand(reduced_im, filter_vec):
    expanded_im = [reduced_im[0]]
    filter_vec *= 2
    x_flag = reduced_im[0].shape[0]% 2
    y_flag = reduced_im[0].shape[1]% 2
    for image in reduced_im[1:]:
        cur_im, x_flag, y_flag = expand_single_pic(image, filter_vec, x_flag, y_flag)
        expanded_im.append(cur_im)
    return expanded_im


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    construct a Laplacian pyramid of a given image.
    :param im: a grayscale image with double values in [0, 1] (e.g. the output of ex1’s
    read_image with the representation set to 1).
    :param max_levels: the maximal number of levels1 in the resulting pyramid
    :param filter_size: the size of the Gaussian filter (an odd scalar that
    represents a squared filter) to be used in constructing the pyramid filter
    :return: pyr, filter_vec
    """
    filter_vec = create_filter(filter_size)
    reduced_im = reduce(im, max_levels, filter_vec)
    expand_im = expand(reduced_im, filter_vec)
    pyr = []
    for image in range(len(expand_im) - 1):
        lap_img = reduced_im[image] - expand_im[image + 1]
        pyr.append(lap_img)
    pyr.append(reduced_im[len(reduced_im) - 1])
    return pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    the reconstruction of an image from its Laplacian Pyramid.
    """
    expanded_pic = np.dot(lpyr[len(lpyr) - 1], coeff[len(lpyr) - 1])
    for i in range(len(lpyr) - 2, -1, -1):
        mult_lp = np.multiply(lpyr[i], coeff[i])
        cur_img = expand_single_pic(expanded_pic, filter_vec, mult_lp.shape[0] % 2, mult_lp.shape[1] % 2)[0]
        expanded_pic = cur_img + mult_lp
    return expanded_pic



def render_pyramid(pyr, levels):
    """
     facilitate the display of pyramids
    """
    x_size, y_size = pyr[0].shape[0], 0
    num_of_levels = min(len(pyr), levels)
    for pic in range(num_of_levels):
        y_size += pyr[pic].shape[1]
    new_img = np.zeros((x_size, y_size))
    cur_position = 0
    for pic in range(num_of_levels):
        img = pyr[pic]
        new_img[0: img.shape[0], cur_position: cur_position + img.shape[1]] = \
            np.true_divide(img - np.min(img), np.max(img))
        cur_position += pyr[pic].shape[1]
    return new_img


def display_pyramid(pyr, levels):
    """
    use render_pyramid to internally render and then display
    the stacked pyramid image using plt.imshow().
    """
    im = render_pyramid(pyr, levels)
    plt.imshow(im, cmap='gray')
    plt.show()

def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    laplacian_im1, filter = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    laplacian_im2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]
    gaussian_m = build_gaussian_pyramid(mask.astype(np.double), max_levels, filter_size_mask)[0]
    mult_pics = np.multiply(gaussian_m, laplacian_im1) + np.multiply((1 - np.array(gaussian_m)), laplacian_im2)
    coeff = np.ones(len(mult_pics))
    sum_levels = laplacian_to_image(mult_pics, filter, coeff)
    im_blend = np.clip(sum_levels, 0, 1)
    return im_blend


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def blending_example1():
    """
    a picture of my nephew with the background of Kashmir.
    The mask was pefectly fitted but i needed to resize the pictures and it screw it up..
    """
    im1 = read_image(relpath('externals/kashmir.jpg'), 2)
    im2 = read_image(relpath('externals/daniel_pic.jpg'), 2)
    mask = read_image(relpath('externals/daniel_mask.jpg'), 2).astype(bool)

    im_blend = np.zeros(im1.shape)
    for color in range(3):
        im_blend[:, :, color] = pyramid_blending(im1[:,:, color], im2[:, :, color], mask, 4, 4, 4)

    pic = plt.figure()
    pic1 = pic.add_subplot(2, 2, 1)
    pic1.imshow(im1, cmap='gray')
    pic2 = pic.add_subplot(2, 2, 2)
    pic2.imshow(im2, cmap='gray')
    pic3 = pic.add_subplot(2, 2, 3)
    pic3.imshow(mask, cmap='gray')
    pic4 = pic.add_subplot(2, 2, 4)
    pic4.imshow(im_blend)
    plt.show()

    return im1, im2, mask, im_blend


def blending_example2():
    """
    floating in the dead sea in the grand canyon
    """
    im1 = read_image(relpath('externals/gc.jpg'), 2)
    im2 = read_image(relpath('externals/ds.jpg'), 2)
    mask = read_image(relpath('externals/floating_mask.jpg'), 1).astype(bool)

    im_blend = np.zeros(im1.shape)
    for color in range(3):
        im_blend[:, :, color] = pyramid_blending(im1[:,:, color], im2[:, :, color], mask, 4, 4, 4)

    pic = plt.figure()
    pic1 = pic.add_subplot(2, 2, 1)
    pic1.imshow(im1, cmap='gray')
    pic2 = pic.add_subplot(2, 2, 2)
    pic2.imshow(im2, cmap='gray')
    pic3 = pic.add_subplot(2, 2, 3)
    pic3.imshow(mask, cmap='gray')
    pic4 = pic.add_subplot(2, 2, 4)
    pic4.imshow(im_blend, cmap='gray')
    plt.show()

    return im1, im2, mask, im_blend

