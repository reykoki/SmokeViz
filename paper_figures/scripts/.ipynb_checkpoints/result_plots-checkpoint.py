from plot_tifs import get_lat_lon, get_datetime_from_fn, get_mesh
import numpy as np
import matplotlib.pyplot as plt

def plot_loss(history):
    train_loss = history['train']
    plt.plot(np.linspace(1,len(train_loss), len(train_loss)), train_loss, 'k')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('training loss')
    plt.show()
    val_loss = history['val']
    plt.plot(np.linspace(1,len(val_loss), len(val_loss)), val_loss, 'k')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('validation loss')
    plt.show()

def plot_test_results(data, labels, pred, fn, save_fig = False):
    RGB = np.dstack([data[0], data[1], data[2]])
    preds = np.dstack([pred[0], pred[1], pred[2]])
    truths = np.dstack([labels[0], labels[1], labels[2]])
    lat, lon = get_lat_lon(fn)
    num_pixels = data.shape[1]
    X, Y = get_mesh(num_pixels)
    colors = ['red', 'orange', 'yellow']
    fig, ax = plt.subplots(1, 2, figsize=(16,8))

    ax[0].imshow(RGB)
    ax[1].imshow(RGB)
    for idx in range(3):
        ax[0].contour(X,Y,truths[:,:,idx],levels =[.99],colors=[colors[idx]])
        ax[1].contour(X,Y,preds[:,:,idx],levels =[.99],colors=[colors[idx]])

    ax[0].set_yticks(np.linspace(0,255,5), np.round(lat,2), fontsize=12)
    ax[0].set_ylabel('latitude (degrees)', fontsize=16)
    ax[0].set_xticks(np.linspace(0,255,5), np.round(lon,2), fontsize=12)
    ax[0].set_xlabel('longitude (degrees)', fontsize=16)
    ax[1].set_yticks([])
    ax[1].set_xticks([])
    ax[0].set_title('analyst annotation',fontsize=18)
    ax[1].set_title('model prediction',fontsize=18)
    plt.suptitle(get_datetime_from_fn(fn), fontsize=18)

    #plt.tight_layout(pad=0)#, h_pad=-.5)
    plt.subplots_adjust(wspace=0)
    if save_fig:
        plt.savefig('./figures/results.png', dpi=300)
    plt.show()
