import numpy as np
import torch
from torchvision.transforms import ToPILImage
from matplotlib import pyplot as plt

def display(subject, ordered):
    def l1_dist(x, y):
        return torch.sum(x - y).item()
    def l2_dist(x, y):
        from math import sqrt
        return sqrt(torch.sum((x - y) ** 2).item())
    def make_panel(list_of_images):
        images = [image[1] for image in list_of_images]
        panel = torch.cat(images, dim=2)
        panel_pil = ToPILImage().__call__(panel)
        return panel_pil

    # sort by distance to the subject
    ordered = sorted(ordered, key=lambda elem: l2_dist(subject[0], elem[0]))
    subject_repeated = [subject for _ in range(10)]
    nearest_10_images = ordered[:10]
    farthest_10_images = ordered[-10:]
    panel_of_subject = make_panel(subject_repeated)
    panel_of_nearest_10 = make_panel(nearest_10_images)
    panel_of_farthest_10 = make_panel(farthest_10_images)
    _img = np.concatenate((panel_of_subject, panel_of_nearest_10, panel_of_farthest_10), axis=0)
    plt.imshow(_img)
    plt.show()