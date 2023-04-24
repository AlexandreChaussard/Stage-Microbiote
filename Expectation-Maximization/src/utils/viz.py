import matplotlib.pyplot as plt
import matplotlib.cm as colormap
import numpy as np
from src.utils.distribution import pdf_gaussian


def plot_2d_gaussians_samples(X, y, subtitle=""):
    fig, axs = plt.subplots()
    fig.suptitle(f"Gaussian samples \n{subtitle}")

    labels = np.unique(y)
    for label in labels:
        indexes = np.where(y == label)
        sub_X = X[indexes]
        axs.plot(sub_X[:, 0], sub_X[:, 1], marker="o", linestyle="", label=str(label))

    axs.legend()
    plt.show()


def plot_2d_gaussians_samples_with_pdf(X, y, mu, sigma, subtitle="", alpha=0.2):
    fig, axs = plt.subplots()
    fig.suptitle(f"Gaussian samples \n{subtitle}")

    labels = np.unique(y)

    # Plot the contours first
    abscisse = np.linspace(np.min(X[:, 0]) - 0.1, np.max(X[:, 0]) + 0.1, 100)
    ordonnee = np.linspace(np.min(X[:, 1]) - 0.1, np.max(X[:, 1]) + 0.1, 100)
    X_contour, Y_contour = np.meshgrid(abscisse, ordonnee)
    pos = np.dstack((X_contour, Y_contour))
    Z_contour = None
    for i in range(len(mu)):
        Z = pdf_gaussian(pos, np.array(mu[i]), sigma[i])
        if Z_contour is not None:
            Z_contour += Z
        else:
            Z_contour = Z

    axs.contourf(X_contour, Y_contour, Z_contour, cmap=colormap.get_cmap('binary'))

    indexes = np.arange(0, len(X))
    np.random.shuffle(indexes)
    label_axes = np.zeros(len(labels))
    for i in indexes:
        x_i = X[i]
        y_i = int(y[i])
        if label_axes[y_i] == 1:
            axs.plot(x_i[0], x_i[1], marker="o", linestyle="", color=f"C{y_i}", alpha=alpha)
        else:
            label_axes[y_i] = 1
            axs.plot(x_i[0], x_i[1], marker="o", linestyle="", color=f"C{y_i}", alpha=alpha, label=y_i)
    axs.legend()
    plt.show()


def plot_1d_gaussian_samples(X, y, n_bins):
    fig, axs = plt.subplots()
    fig.suptitle("Gaussian samples")

    labels = np.unique(y)
    for label in labels:
        indexes = np.where(y == label)
        axs.hist(X[indexes], bins=n_bins, label=str(label))

    axs.legend()
    plt.show()


def plot_1d_gaussian_samples_with_pdf(X, y, mu_list, sigma_list, n_bins, subtitle):
    fig, axs = plt.subplots()
    fig.suptitle(f"Gaussian samples with pdf \n{subtitle}")

    labels = np.unique(y)
    for label in labels:
        indexes = np.where(y == label)
        axs.hist(X[indexes], bins=n_bins, alpha=0.5, color="C0")

    abscisse = np.linspace(np.min(X), np.max(X), 200).reshape([-1, 1])
    for i in range(len(mu_list)):
        mu = mu_list[i]
        sigma = sigma_list[i]
        axs.plot(abscisse, pdf_gaussian(abscisse, mu, sigma), color="C0")

    plt.show()


def plot_individual_microbiota(images, n_rows, n_cols):
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 9))
    i = 0
    j = 0
    for image in images:
        axs[i][j].imshow(image, cmap='Blues')

        j += 1
        if j % n_cols == 0:
            i += 1
            j = 0

    for i in range(len(axs)):
        for j in range(len(axs[i])):
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
            axs[i][j].axis('off')

    fig.suptitle("Image representation of microbiota abundance")
    plt.show()
