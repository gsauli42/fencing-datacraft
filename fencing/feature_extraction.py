import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def extract_features_person_1frame(frame, out, label="p1"):
    """Extract features from a single frame for 1 person

    Parameters
    ----------
    frame : ndarray of shape (n_frames, 3)
       input data
    out : dict
       dictionary where to store the results
    label : str
       label of the person
    """
    right_leg = np.nanmean(frame[[11, 22, 23, 24]][:, :2], axis=0)
    left_leg = np.nanmean(frame[[14, 19, 20, 21]][:, :2], axis=0)
    mhip = np.nanmean(frame[[8, 9, 12]][:, :2], axis=0)

    # return (right_leg - left_leg)[0]
    out[f"leg_distance_{label}"] = np.sqrt(((right_leg - left_leg) ** 2).sum())
    out[f"leg_distance_dx_{label}"] = np.abs((right_leg - left_leg)[0])
    out[f"leg_distance_dy_{label}"] = np.abs((right_leg - left_leg)[1])

    tmp = right_leg - mhip
    out[f"left_leg_angle_{label}"] = np.abs(np.arctan2(tmp[0], tmp[1]) * 180 / np.pi)
    tmp = left_leg - mhip
    out[f"right_leg_angle_{label}"] = np.abs(np.arctan2(tmp[0], tmp[1]) * 180 / np.pi)
    return


def extract_features_1frame(frame):
    """Extract features from 1 frame containing 2 people

    Parameters
    ----------
    frame : ndarray of shape (n_people=2, n_frames, 3)
       input data
    out : dict
       dictionary where to store the results
    label : str
       label of the person
    """
    out = {}
    frame[frame == 0] = np.nan
    extract_features_person_1frame(
        frame[0], out, "p0",
    )
    extract_features_person_1frame(frame[1], out, "p1")
    mhip_0 = np.nanmean(frame[0, [8, 9, 12]][:, :2], axis=0)
    mhip_1 = np.nanmean(frame[1, [8, 9, 12]][:, :2], axis=0)
    out["mhip_distance_x"] = np.abs((mhip_0 - mhip_1))[0]
    out["mhip_distance_x_p0"] = mhip_0[0]
    out["mhip_distance_x_p1"] = mhip_1[0]
    return out


def extract_features(frame_sequence):
    """Extract features from a sequence of frames

    Parameters
    ----------
    frame_sequence : sequence of ndarray of shape (n_people=2, n_frames, 3)
       input data

    Returns
    -------
    df_fr: pd.DataFrame
       extracted features
    """

    df_fe = pd.DataFrame(
        [extract_features_1frame(frame) for frame in frame_sequence]
    ).round(2)

    df_fe["mhip_speed_x_diff"] = savgol_filter(
        df_fe["mhip_distance_x"].values, 7, polyorder=3, deriv=1
    )
    df_fe["mhip_speed_x_p0"] = savgol_filter(
        df_fe["mhip_distance_x_p0"].values, 7, polyorder=3, deriv=1
    )
    df_fe["mhip_speed_x_p1"] = savgol_filter(
        - df_fe["mhip_distance_x_p1"].values, 7, polyorder=3, deriv=1
    )
    df_fe["idx"] = df_fe.index.values
    return df_fe
