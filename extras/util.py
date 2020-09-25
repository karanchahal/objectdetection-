import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, box, fill=False):
    ex1 = box[0]
    ey1 = box[1]
    ex2 = box[2]
    ey2 = box[3]
    rect = patches.Rectangle((ex1,ey1),abs(ex1 - ex2),abs(ey1 - ey2),linewidth=1, edgecolor='r',fill=fill, facecolor='r')
    ax.add_patch(rect)

def vis(sample):
    ''' Assumes batch size of 4'''
    sample = sample.permute(0,2,3,1)
    fig, axs = plt.subplots(2,2)
    axs[0,0].imshow(sample[0])
    axs[0,1].imshow(sample[1])
    axs[1,0].imshow(sample[2])
    axs[1,1].imshow(sample[3])
    plt.show()

def visDet(sample, target):
    ''' Assumes batch size of 4'''
    sample = sample.permute(0,2,3,1)
    fig, axs = plt.subplots(2,2)

    axs[0,0].imshow(sample[0])
    for t in target[0]['bounding_box']:
        draw_box(axs[0,0], t)

    axs[0,1].imshow(sample[1])
    for t in target[1]['bounding_box']:
        draw_box(axs[0,1], t)

    axs[1,0].imshow(sample[2])
    for t in target[2]['bounding_box']:
        draw_box(axs[1,0], t)

    axs[1,1].imshow(sample[3])
    for t in target[3]['bounding_box']:
        draw_box(axs[1,1], t)

    plt.show()