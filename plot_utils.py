import matplotlib.pyplot as plt



def plot_heatmap(matrix, x_figsize, y_figsize, title):
    fig, ax = plt.subplots(figsize=(x_figsize, y_figsize))
    cax = ax.imshow(matrix, aspect="auto")

    fig.colorbar(cax)

    ax.set_title(title)

    plt.tight_layout()
    plt.show()
