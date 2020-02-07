import json
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from ipywidgets import interactive


def visualize_features(df_fe):
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    #x = np.linspace(0, 5, len(leg_dist))
    df_fe.plot(y="leg_distance_p0", color='r', ax=ax[0])
    df_fe.plot(y="leg_distance_p1", color='b', ax=ax[0])
    #df_fe.plot(y="right_leg_angle_p0", color='r', ax=ax[1])
    #df_fe.plot(y="right_leg_angle_p1", color='b', ax=ax[1])
    df_fe.plot(y="mhip_distance_x", color='k', ax=ax[1])
    df_fe.plot(y="mhip_speed_x_diff", ls='dashed', color='k', ax=ax[2])
    df_fe.plot(y="mhip_speed_x_p0", color='r', ax=ax[2])
    df_fe.plot(y="mhip_speed_x_p1", color='b', ax=ax[2])
    ax[0].legend()
    plt.show()


def load_video(video_path='../data/video.mp4'):
    """Load initial video

    Parameters
    ----------
    video_path : str
       path to the video

    Returns
    -------
    cap: VideoCapture
        video loaded by cv2
    """
    cap = cv2.VideoCapture(video_path)
    return cap


def get_frame_from_video(cap, frame_number):
    # Set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    return frame

#
# def plot_key_points(X, ax):
#     mask = X[0].mean(axis=1) > 1
#     ax.plot(X[0][mask, 0], X[0][mask, 1], 'ro')
#     mask = X[1].mean(axis=1) > 1
#     ax.plot(X[1][mask, 0], X[1][mask, 1], 'bo')
#     ax.set_ylim(230, 0)
def plot_key_points(X, ax, annotate=False):
    mask = X[0].mean(axis=1) > 1
    ax.plot(X[0][mask, 0], X[0][mask, 1], 'ro')
    if annotate:
        for idx in np.nonzero(mask)[0]:
            ax.text(X[0][idx, 0], X[0][idx, 1], s=str(idx))
    mask = X[1].mean(axis=1) > 1
    ax.plot(X[1][mask, 0], X[1][mask, 1], 'bo')
    if annotate:
        for idx in np.nonzero(mask)[0]:
            ax.text(X[1][idx, 0], X[1][idx, 1], s=str(idx))
    ax.set_ylim(280, 0)

# def plot_key_point_sequence(idx):
#     fig, ax = plt.subplots()
#     plot_key_points(df_kp[int(idx)], ax)
#     plt.show()
def plot_key_point_sequence(idx):
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_key_points(df_kp[int(idx)], ax, annotate=True)
    plt.show()


def setup_figure(N_row=25, N_col=3, fig_width=20, fig_height=50):
    f, axs = plt.subplots(N_row, N_col, sharex='col', sharey='row',
                          gridspec_kw={'hspace': 0, 'wspace': 0},
                          constrained_layout=True)

    f.set_figwidth(fig_width)
    f.set_figheight(fig_height)
    return f, axs


def plot_all_from_clip(df_kp, start, axs, cap):
    idx = 0
    N_row = len(axs)
    N_col = len(axs[0])
    for row in range(N_row):
        for col in range(N_col):
            ax = axs[row][col]
            try:
                ax.imshow(get_frame_from_video(cap, int(start + idx)))
                plot_key_points(df_kp[idx], ax)
                ax.set_aspect("equal")
                ax.text(.9,.8,str(idx),
                        color="white", size=20, fontweight="bold",
                        horizontalalignment='center',
                        transform=ax.transAxes)
            except Exception as e:
                print("Error:", e)
                pass
            idx += 1
    plt.tight_layout()
    plt.show()
