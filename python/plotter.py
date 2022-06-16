from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import manifold
import matplotlib.patheffects as PathEffects
import matplotlib.gridspec as gridspec
from PIL import Image
import imageio
from collections import OrderedDict

sns.set(style="white", font_scale=1.5, palette="muted", color_codes=True, rc={"lines.linewidth": .5, 'legend.fontsize': 16,
                                                                              'xtick.labelsize': 16, 'axes.labelsize': 18})
#sns.set(style="white", palette="muted", color_codes=True, rc={"lines.linewidth": .5})
LINESTYLES = [
  (0, ()),
  (0, (5, 10)),
  (0, (5, 5)),
  (0, (3, 5, 1, 5, 1, 5)),
  (0, (3, 1, 1, 1, 1, 1))
]

import matplotlib.patheffects as PathEffects
import matplotlib.gridspec as gridspec

def visualize_hash_codes(inX, inY, label_names=None, filename=None, figsize=(10, 10), title=None):
    if type(inY) != np.ndarray:
        inY = np.asarray(inY)
    labels = sorted(list(set(inY)))
    labelCnt = len(labels)
    print('Number of labels: {}'.format(labelCnt))
    #palette = np.array(sns.color_palette("hls", labelCnt))
    #palette = np.array(['red', 'green', 'blue', 'orange'])
    palette = np.array(sns.color_palette("bright", labelCnt))

    indexedY = [labels.index(e) for e in inY]

    f = plt.figure(figsize=figsize)
    ax = plt.subplot(aspect='equal')
    f.patch.set_visible(False)
    #ax.axis('off')
    
    sc = ax.scatter(inX[:, 0], inX[:, 1], lw=0, s=30, alpha=0.8,
                  c=palette[indexedY])
    #plt.xlim(np.min(inX[:, 0]) - 0.01, np.max(inX[:, 0]) + 0.01)
    #plt.ylim(np.min(inX[:, 1]) - 0.01, np.max(inX[:, 1]) + 0.01)
    #plt.xlim(-1, 1)
    #plt.ylim(-1, 1)
    #ax.axis('off')
    ax.axis('tight')
    ax.axhline(y=0, color='black', linestyle='dashed', linewidth=3)
    ax.axvline(x=0, color='black', linestyle='dashed', linewidth=3)
    if title is not None:
        ax.set_title(title, fontsize=20)
    #ax.set_xticks([-1,0,1])
    #ax.set_yticks([-1,0,1])
  # We add the labels for each digit.
    txts = []
    for i in labels:
        # Position of each label.
        #xtext, ytext = np.median(inX[inY == i, :], axis=0)
        xtext, ytext = np.mean(inX[inY == i, :], axis=0)
        label_name = str(i)
        if label_names is not None:
            label_name = label_names[i]
        txt = ax.text(xtext, ytext, label_name, fontsize=24, color=palette[i])
        txt.set_path_effects([
          PathEffects.Stroke(linewidth=5, foreground="w"),
          PathEffects.Normal()])
        txts.append(txt)

    if filename is not None:
        plt.savefig(filename, dpi=200)
        
# plot_embedding(torch.tanh(s_tst_u).cpu().numpy(), s_tst_label.argmax(axis=1),
#                figsize=(4,4))
# plot_embedding(torch.tanh(s_tst_u).cpu().numpy(), s_tst_label.argmax(axis=1),
#                figsize=(4,4))

def savefig(filename, dpi=300):
  plt.tight_layout()
  plt.savefig(filename, dpi=dpi)
def tsne_embed_and_plot(origX, inY, filename=None, figsize=(10,10),
                        n_iters=1000, lr=10, perplexity=50, verbose=2, n_iter_without_progress=100):
  inX = tsne_embed(origX,
                   n_iters=n_iters, lr=lr, perplexity=perplexity, verbose=verbose,
                   n_iter_without_progress=n_iter_without_progress)

  plot_embedding(inX, inY, filename, figsize)
  return inX

def tsne_embed(origX, n_iters=1000, lr=10, perplexity=50, verbose=2, n_iter_without_progress=100):
  inX = manifold.TSNE(
    n_components=2, verbose=verbose, n_iter=n_iters, learning_rate=lr, perplexity=perplexity,
    n_iter_without_progress=n_iter_without_progress
  ).fit_transform(origX)

  return inX

def plot_embeddings(inXs, inYs, titles = None, nColsPerRow=3, filename=None, figsize=(10, 10), patterns=None):
  if titles is None:
    titles = [None for _ in inXs] #create dumpy fig_labels
  n = len(inXs)
  r = int(np.ceil(n * 1.0 / 3))

  fig = plt.figure(figsize=figsize)
  # fig, axes = plt.subplots(r, nColsPerRow, figsize=figsize)

  for idx, (inX, inY, title) in enumerate(zip(inXs, inYs, titles)):
    ax = fig.add_subplot(r, nColsPerRow, idx+1)
    plot_one_embedding(inX, inY, ax, title=title)

  if filename is not None:
    plt.savefig(filename, dpi=300)

def plot_one_embedding(inX, inY, ax, title=None, verbose=0):
  if type(inY) != np.ndarray:
    inY = np.asarray(inY)
  labels = sorted(list(set(inY)))
  labelCnt = len(labels)
  if verbose>0:
    print('Number of labels: {}'.format(labelCnt))

  palette = np.array(sns.color_palette("hls", labelCnt))

  indexedY = [labels.index(e) for e in inY]

  ax.set_aspect('equal')
  sc = ax.scatter(inX[:, 0], inX[:, 1], lw=0, s=15,
                  c=palette[indexedY])
  plt.xlim(np.min(inX[:, 0]) - 0.01, np.max(inX[:, 0]) + 0.01)
  plt.ylim(np.min(inX[:, 1]) - 0.01, np.max(inX[:, 1]) + 0.01)
  ax.axis('off')
  ax.axis('tight')

  # We add the labels for each digit.
  txts = []
  for i in labels:
    # Position of each label.
    xtext, ytext = np.median(inX[inY == i, :], axis=0)
    txt = ax.text(xtext, ytext, str(i), fontsize=24)
    txt.set_path_effects([
      PathEffects.Stroke(linewidth=5, foreground="w"),
      PathEffects.Normal()])
    txts.append(txt)

  if title is not None:
    ax.set_title(title, fontsize=20)

def plot_embedding(inX, inY, label_names=None, filename=None, figsize=(10, 10), title=None):
  if type(inY) != np.ndarray:
    inY = np.asarray(inY)
  labels = sorted(list(set(inY)))
  labelCnt = len(labels)
  print('Number of labels: {}'.format(labelCnt))
  palette = np.array(sns.color_palette("hls", labelCnt))

  indexedY = [labels.index(e) for e in inY]

  f = plt.figure(figsize=figsize)
  ax = plt.subplot(aspect='equal')
  sc = ax.scatter(inX[:, 0], inX[:, 1], lw=0, s=30, alpha=0.8,
                  c=palette[indexedY])
  plt.xlim(np.min(inX[:, 0]) - 0.01, np.max(inX[:, 0]) + 0.01)
  plt.ylim(np.min(inX[:, 1]) - 0.01, np.max(inX[:, 1]) + 0.01)
  ax.axis('off')
  ax.axis('tight')
  if title is not None:
    ax.set_title(title, fontsize=20)

  # We add the labels for each digit.
  txts = []
  for i in labels:
    # Position of each label.
    #xtext, ytext = np.median(inX[inY == i, :], axis=0)
    xtext, ytext = np.mean(inX[inY == i, :], axis=0)
    label_name = str(i)
    if label_names is not None:
        label_name = label_names[i]
    txt = ax.text(xtext, ytext, label_name, fontsize=24, color=palette[i])
    txt.set_path_effects([
      PathEffects.Stroke(linewidth=5, foreground="w"),
      PathEffects.Normal()])
    txts.append(txt)

  if filename is not None:
    savefig(filename, dpi=300)

def plot_count_boxplots(df, groupCols, filename=None, title=None):
  counts = [row['count'] for row in df.groupby(groupCols).count().collect()]
  sns.boxplot(counts, orient='v')

  if title is not None:
    plt.title(title)

  if filename is not None:
    plt.savefig(filename, dpi=300)

def plot_labeled_hashed_points(X_hashes, y):
  X_embedded = manifold.TSNE(n_components=2, verbose=1).fit_transform(X_hashes)
  plot_embedding(X_embedded, y)

def plot_activation_histograms(activation_sets, titles=None, nColsPerRow=3, bins=10, filename=None, figsize=(5,3)):
  if titles is None:
    titles = [None for _ in activation_sets] #dummy title
  rows = int(np.ceil(len(activation_sets) * 1.0 / nColsPerRow))
  fig = plt.figure(figsize=figsize)
  for idx, (activations, title) in enumerate(zip(activation_sets, titles)):
    ax = fig.add_subplot(rows, nColsPerRow, idx+1)
    plot_one_activation_histogram(ax, activations, bins=bins, title=title)

  if filename is not None:
    plt.savefig(filename, dpi=300)

def plot_one_activation_histogram(ax, activations, bins=10, title=None):
  for i in range(activations.shape[1]):
    sns.distplot(activations[:, i], kde=False, bins=bins, ax=ax, color='blue')

  ax.set_xlim(0, 1)
  ax.set_xticklabels([])
  ax.set_yticklabels([])
  ax.set_title(title if title else 'Activation Distribution')

def plot_activation_histogram(activations, bins=10, title=None, filename=None, figsize=(5,3), aggregate=False):
  fig, ax = plt.subplots(1, 1, figsize=figsize)
  if aggregate:
    #sns.distplot(np.reshape(activations, [-1]), kde=False, bins=bins, ax=ax, color='blue')
    ax.hist(np.reshape(activations, [-1]), bins=bins, color='blue')
  else:
    for i in range(activations.shape[1]):
      sns.distplot(activations[:, i], kde=False, bins=bins, ax=ax, color='blue')
  # sns.distplot(
  #   activations, kde=False, bins=bins,
  #   color=sns.color_palette("Set1", n_colors=8, desat=.5)[1],
  #   ax=ax
  # )

  ax.set_xlim(0, 1)
  #ax.set_xticklabels([])
  ax.set_yticklabels([])
  ax.set_title(title if title else 'Activation Distribution')

  if filename is not None:
    savefig(filename, dpi=300)
  else:
    plt.show()
  plt.close()

def plot_losses(train_losses, valid_losses, filename=None):
  fig, ax = plt.subplots(1, 1)
  n = 10 if len(train_losses) > 10 else 1
  x = range(2, len(train_losses) + 1)
  spacing = len(x) / n
  sns.pointplot(x=x, y=train_losses[1:], color='blue', label='Train', ax=ax)
  sns.pointplot(x=x, y=valid_losses[1:], color='red', label='Valid', ax=ax)
  ax.set_xticklabels([str(i) if (i % spacing == 0 or i == 2) else '' for i in x])

  if filename is not None:
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_distance_distributions(nDistances, rDistances, title=None, filename=None):
  for i in range(1, min(10, nDistances.shape[1])):
    ax = sns.kdeplot(nDistances[:, i], label='nearest_{}'.format(i))
  ax = sns.kdeplot(rDistances[0], color='black', label='random')
  for i in range(1, len(rDistances)):
    ax = sns.kdeplot(rDistances[i], color='black')
  ax.legend(loc=0)
  ax.set_xlabel('Distance')
  ax.set_ylabel('Density')
  ax.set_title('Distance distribution of nearest vs random' if not title else title)
  ax.set_xlim(0, 1.45)
  plt.tight_layout()
  if filename is not None:
    plt.savefig(filename)
    plt.close()

def plot_distance_distribution(d, k=1, title=None):
  ax = sns.distplot(
    d[:, k], bins=30
  )
  ax.set_xlabel('Distance')
  ax.set_ylabel('Density')
  ax.set_title('Distance distribution of 1-nearest neigbors' if not title else title)
  plt.tight_layout()


def plot_images(samples, filename, title=None):
  n = samples.shape[0]
  fig = plt.figure(figsize=(8, n))
  gs = gridspec.GridSpec(1, n)
  gs.update(wspace=0.05, hspace=0.05)

  for i, sample in enumerate(samples):
    ax = plt.subplot(gs[i])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
  if title:
    plt.title(title)
  if filename is not None:
    plt.savefig(filename)
    plt.close()
  else:
    plt.show()
    plt.close()


def merge_figures(filenames, ncols, output_filename):
  """Merge figures into 1"""
  n = len(filenames)
  imgs = [Image.open(fn) for fn in filenames]
  nrows = np.ceil(n * 1.0 / ncols).astype(int)
  h, w, c = imgs[0].width, imgs[0].height, len(imgs[0].mode)
  padded_white_images = np.zeros(shape=(w, h, c), dtype=np.uint8)
  if nrows * ncols > n:
    imgs += [padded_white_images for _ in range(n, ncols * nrows)]

  img_merge = []
  for ridx in range(nrows):
    img_merge.append(np.hstack(imgs[ridx * ncols:(ridx + 1) * ncols]))
  img_merge = np.vstack(img_merge)
  img_merge = Image.fromarray(img_merge)
  img_merge.save(output_filename)

def create_gif(output_filename, filenames, duration=0.2):
  images = []
  for filename in filenames:
    images.append(imageio.imread(filename))
  imageio.mimsave(output_filename, images, duration=duration)

def plot_lines(xx, yy, legend=None, filename=None, figsize=(5,3), linestyles=LINESTYLES, xlabel=None, ylabel=None):
  fig, ax = plt.subplots(figsize=(8, 5))
  markers = ['o', 'X', 'D', '*', 'h', 'x']
  if type(xx) != list:
    xx = [xx for _ in yy] #replicate the x-axis

  palettes = sns.color_palette("bright", len(xx))
  xlims = (np.min([np.min(x) for x in xx]), np.max([np.max(x) for x in xx]))
  ylims = (np.min([np.min(y) for y in yy]), np.max([np.max(y) for y in yy]))
  lines = []
  for idx, (x, y, p) in enumerate(zip(xx, yy, palettes)):
    # if legend is None:
    #   sns.pointplot(x, y, color=p, scale=3.0, ax=ax)
    # else:
    #   sns.pointplot(x, y, color=p, scale=3.0, ax=ax, label=legend[idx])
    line, = ax.plot(x, y, color=p, linewidth=3.0, linestyle=linestyles[idx], marker='o')
    lines.append(line)
  if xlabel:
    ax.set_xlabel(xlabel)
  if ylabel:
    ax.set_ylabel(ylabel)
  ax.set_xlim(xlims[0], xlims[1])
  ax.set_ylim(ylims[0], ylims[1])
  ax.set
  if legend is not None:
    ax.legend(lines, legend, loc=(0.01,0.01))
  if filename is not None:
    savefig(filename)