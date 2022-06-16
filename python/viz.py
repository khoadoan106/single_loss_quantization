from matplotlib import pyplot as plt
from matplotlib import animation

import torchvision.utils as vutils

import numpy as np
from scipy.misc import imsave

def plot_torch_images(images, n=16, nrow=16, filename=None, figsize=(10,10), normalize=False, title=None, padding=0, pad_value=0, dpi=150):
    """Plot torch images from a tensor.
    :param n: number of images to plot from the tensor.
    :param nrow: number of images per row.
    :param filename: filename to save.
    :param figsize: figure size.
    :param normalize: whether to normalize the pixel values.
    :param title: title of the plot.
    :param padding: amount of padding between images.
    :param pad_value: padding color value.
    :param dpi: saved dpi.
    """
    grid_images = vutils.make_grid(images.detach().clone()[:n], padding=padding, pad_value=pad_value, normalize=normalize, nrow=nrow).detach().cpu()
    if figsize:
        plt.figure(figsize=figsize)
    else:
        plt.figure()
    plt.axis("off")
    plt.imshow(np.transpose(grid_images,(1,2,0)))
    if title:
        plt.title(title)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=dpi)
        plt.close()
    else:
        plt.show()

def save_torch_images(images, n=16, nrow=16, filename=None, normalize=False, padding=0, pad_value=0):
    """Save torch images from a tensor. 
    This is different from plot_torch_images where we use imsave instead of pyplot.
    :param n: number of images to plot from the tensor.
    :param nrow: number of images per row.
    :param filename: filename to save.
    :param normalize: whether to normalize the pixel values.
    :param padding: amount of padding between images.
    :param pad_value: padding color value.
    """
    grid_images = vutils.make_grid(images.detach().clone()[:n], padding=padding, pad_value=pad_value, normalize=normalize, nrow=nrow).detach().cpu()
    imsave(filename, np.transpose(grid_images,(1,2,0)))
    
def plot_image(image, title=None, figsize=(8, 8), filename=None):
    """Plot a single image"""
    plt.figure(figsize=figsize)
    # Plot the real images
    plt.axis("off")
    plt.title(title)
    plt.imshow(np.transpose(image, (1, 2, 0)))

    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    else:
        plt.show()
        
def plot_images(image_sets, titles, figsize=(8, 8), ncols = 1, nrows = 1, filename=None):
    """Combine images from list of numpy images into a grid."""
    n = len(image_sets)

    assert n <= ncols * nrows

    plt.figure(figsize=figsize)

    for idx, (images, title) in enumerate(zip(image_sets, titles)):
        # Plot the real images
        plt.subplot(nrows, ncols, idx % ncols + 1)
        plt.axis("off")
        plt.title(title)
        plt.imshow(np.transpose(images, (1, 2, 0)))

    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    else:
        plt.show()

def plot_animation(img_list, filename):
    """Plot animation of a list of images"""
    Writer = animation.writers['html']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    # Visualization progression
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    ani.save(filename, writer=writer)