from matplotlib import pyplot as plt
from praudio import utils


def plot_confusion_matrix(confusion_matrix, size, save_path=''):
    fig, ax = plt.subplots(figsize=(size/2, size/2))
    ax.matshow(confusion_matrix, cmap=plt.cm.Blues, alpha=0.3)

    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(x=j, y=i, s=confusion_matrix[i, j],
                    va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)

    fig.tight_layout()
    plt.show()
    plt.draw()

    if save_path:
        utils.create_dir_hierarchy(save_path)
        fig.savefig(f'{save_path}/confusion_matrix.jpg',
                    dpi=300)

    plt.close()


def plot_history(history, save_path=''):
    """Plots accuracy/loss for training/validation set as a function of the epochs

    :param history: Training history of model
    :return:
    """

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="accuracy")
    axs[0].plot(history.history['val_accuracy'], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")
    axs[0].margins(x=0)
    axs[0].grid()

    # create loss subplot
    axs[1].plot(history.history["loss"], label="loss")
    axs[1].plot(history.history['val_loss'], label="val_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss evaluation")
    axs[1].margins(x=0)
    axs[1].grid()

    fig.tight_layout()
    plt.show()
    plt.draw()

    if save_path:
        utils.create_dir_hierarchy(save_path)
        fig.savefig(f'{save_path}/train_history.jpg',
                    dpi=300)

    plt.close()
