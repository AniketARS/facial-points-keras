import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def get_data(idx, key_pts_frame):

    if idx > key_pts_frame.shape[0]:
        raise ValueError('idx should be less than', key_pts_frame.shape[0])

    image_name = key_pts_frame.iloc[idx, 0]
    image = mpimg.imread(os.path.join('./data/training', image_name))

    key_pts = key_pts_frame.iloc[idx, 1:].to_numpy()
    key_pts = key_pts.astype('float').reshape(-1, 2)

    return image, key_pts

def plot_keypoints(image, key_pts):
    plt.imshow(image)
    plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, c='m', marker='.')

def show_keypoints(idx, key_pts_frame):
    image, key_pts = get_data(idx, key_pts_frame=key_pts_frame)
    plot_keypoints(image, key_pts=key_pts)
    plt.show()
