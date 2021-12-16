"""
AUTHOR : Salil MARATH PONMADOM
-------------------------------------------------
The Module contains Helper functions which facilitates the Video Reconstructor class.
"""

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

from typing import List, Any, Dict, Tuple
import imagehash
from PIL import Image


def get_frames_from_video(file_path: str) -> Any:
    """
    Creates a new directory inside the project datasets directory and stores all the frames extracted.

    Parameters
    -----------
    file_path: str
        The path which specifies the Corrupted Video.

    Returns
    --------
    Stores all the frames extracted from the video, creates and stores the frames in a new directory
    and returns the path to the frame directory.

    """

    video = cv2.VideoCapture(file_path)
    try:
        if not os.path.exists('../datasets/vedio_frames_test'):
            os.makedirs('../datasets/vedio_frames_test')

    except OSError:
        print('Error: Creating directory of data')

    currentframe = 0
    while (True):

        success, frame = video.read()

        if success:
            FRAME_DIRECTORY_PATH = '../datasets/vedio_frames_test/'
            name = FRAME_DIRECTORY_PATH + 'frame' + str(currentframe) + '.jpg'
            cv2.imwrite(name, frame)
            currentframe += 1
        else:
            break

    video.release()
    cv2.destroyAllWindows()
    return FRAME_DIRECTORY_PATH


def display_frames(frame_list: List[np.array()]) -> Any:
    """
    The function to visualize list of all frames passed as a list via Subplot.
    Helpful to see the cohesion between adjacent video frames to switch similarity methods.

    Parameters
    -----------
    frame_list: List[np.array]
    The list of array images of Frames to be Visualized.

    Returns
    --------
    Displays the Subplot containing all the Images passed.

    """

    fig = plt.figure(figsize=(80, 80))
    rows = round(len(frame_list) / 9)
    columns = round(len(frame_list) / 9)

    for i in range(1, len(frame_list) + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(frame_list[i - 1])


def get_sorted_frames(frame_directory: Dict[str, Any], distance_metrics: List[Tuple]) -> List[np.array()]:
    """
    The function to generate frame index along with array image of frames and frame index along with
    computed frame histogram values as Dictionaries

    Parameters
    -----------
    frame_directory: Dict[str, Any]
        The dictionary containing frame name as key and array image of frame as values.

    distance_metrics: List[Tuple]
        The List of tuples sorted, where each tuple has frame name and corresponding
            similarity score with respect to the image reference chosen.

    Returns
    --------
    The sorted List of array images of frames.

    """
    sorted_idx = []
    for result in distance_metrics:
        sorted_idx.append(result[1])
    sorted_frames = []
    for idx in sorted_idx:
        sorted_frames.append(frame_directory[idx])
    return sorted_frames


def eliminate_outlier_frame(frame_directory: Dict[str, Any], distance_metrics: List[Tuple],
                            hash_threshold: int = 20) -> List[np.array]:
    """
    The function which computes the image hash difference between adjacent frames in the sorted list
    and eliminates the outliers which has a hash difference higher than the passed threshold.

    Parameters
    -----------
    frame_directory: Dict[str, Any]
        The dictionary containing frame name as key and array image of frame as values.

    distance_metrics: List[Tuple]
        The List of tuples sorted, where each tuple has frame name and corresponding
            similarity score with respect to the image reference chosen.

    Returns
    --------
        The sorted List of array images of frames by eliminating the outlier frames.

    """
    hash_difference = []
    result_frames = []
    for result in distance_metrics:
        result_frames.append(frame_directory[result[1]])

    for i in range(len(result_frames) - 1):
        hash_value = imagehash.average_hash(Image.fromarray(result_frames[i]))
        hash_value_next = imagehash.average_hash(Image.fromarray(result_frames[i + 1]))
        hash_diff = hash_value - hash_value_next
        hash_difference.append(hash_diff)

    outlier_hashes = [idx for idx, _hash in enumerate(hash_difference) if _hash >= hash_threshold]
    clean_frames = result_frames[0:outlier_hashes[0] + 1]

    return clean_frames



