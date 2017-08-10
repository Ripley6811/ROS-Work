import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from pcl_helper import *


def rgb_to_hsv(rgb_list):
    rgb_normalized = [1.0*rgb_list[0]/255, 1.0*rgb_list[1]/255, 1.0*rgb_list[2]/255]
    hsv_normalized = matplotlib.colors.rgb_to_hsv([[rgb_normalized]])[0][0]
    return hsv_normalized


def compute_color_histograms(cloud, using_hsv=False, bins=32, range=(0, 256)):

    # Compute histograms for the clusters
    point_colors_list = []

    # Step through each point in the point cloud
    for point in pc2.read_points(cloud, skip_nans=True):
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
        else:
            point_colors_list.append(rgb_list)

    # Populate lists with color values
    channel_1_vals = []
    channel_2_vals = []
    channel_3_vals = []

    for color in point_colors_list:
        channel_1_vals.append(color[0])
        channel_2_vals.append(color[1])
        channel_3_vals.append(color[2])
    
    # Compute histograms
    ch1_hist = np.histogram(channel_1_vals, bins=bins, range=range, density=True)[0]
    ch2_hist = np.histogram(channel_2_vals, bins=bins, range=range, density=True)[0]
    ch3_hist = np.histogram(channel_3_vals, bins=bins, range=range, density=True)[0]

    # Concatenate and normalize the histograms
    hist_features = np.concatenate((ch1_hist, ch2_hist, ch3_hist))
    normed_features = hist_features.astype('float') / np.sum(hist_features)
    
    #print type(normed_features), normed_features.dtype, normed_features
    #fig = plt.figure(figsize=(12,6))
    #plt.plot(normed_features)
    #plt.title('HSV Feature Vector', fontsize=30)
    #plt.tick_params(axis='both', which='major', labelsize=20)
    #fig.tight_layout()
    #plt.show()

    return normed_features 
    
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hists = [np.histogram(img_hsv[:,:,i], bins=nbins, range=bins_range)[0] for i in [0,1,2]]
    hist_features = np.concatenate((hists[0], hists[1], hists[2]))
    norm_features = hist_features / np.sum(hist_features)
    return norm_features
    """


def compute_normal_histograms(normal_cloud, bins=32, range=(-1.0, 1.0)):
    norm_x_vals = []
    norm_y_vals = []
    norm_z_vals = []

    for norm_component in pc2.read_points(normal_cloud,
                                          field_names = ('normal_x', 'normal_y', 'normal_z'),
                                          skip_nans=True):
        norm_x_vals.append(norm_component[0])
        norm_y_vals.append(norm_component[1])
        norm_z_vals.append(norm_component[2])

    # Compute histograms of normal values (just like with color)
    nx_hist = np.histogram(norm_x_vals, bins=bins, range=range, density=True)
    ny_hist = np.histogram(norm_y_vals, bins=bins, range=range, density=True)
    nz_hist = np.histogram(norm_z_vals, bins=bins, range=range, density=True)

    # Concatenate and normalize the histograms
    hist_features = np.concatenate((nx_hist[0], ny_hist[0], nz_hist[0]))
    normed_features = hist_features / np.sum(hist_features)

    #print type(normed_features), normed_features.dtype, normed_features
    return normed_features

    