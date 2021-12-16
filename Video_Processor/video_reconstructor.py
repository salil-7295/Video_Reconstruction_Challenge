"""
AUTHOR : Salil MARATH PONMADOM
-------------------------------------------------
The class which facilitates the video reconstruction pipeline.
"""


import cv2
import numpy as np
import glob
from typing import Any, Dict, List, Tuple
from scipy.spatial import distance as dist
from video_utilities import *


class video_Reconstructor():

    def __init__(self, video_path) -> None:

        self.video_path = video_path
        self.frame_path = get_frames_from_video(video_path)

    def get_image_hist_dictionary(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        The function to generate frame index along with array image of frames and frame index along with
        computed frame histogram values as Dictionaries

        Parameters
        -----------

        Returns
        --------
        frame_directory: Tuple[Dict[str, Any]
            The dictionary containing frame name as key and array image of frame as values.

        histogram_directory : Dict[str, Any]
            The dictionary containing frame name as key and corresponding computed histogram
            as values.

        """

        frame_directory = {}
        histogram_directory = {}
        path = self.frame_path
        for imagePath in glob.glob(path + "/*.jpg"):
            filename = imagePath[imagePath.rfind("/") + 1:]
            image = cv2.imread(imagePath)
            frame_directory[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            histogram_directory[filename] = hist
        return frame_directory, histogram_directory

    @staticmethod
    def frame_sorting_opencv(frame_directory: Dict[str, Any], reference_image: str,
                             metric_choice=cv2.HISTCMP_BHATTACHARYYA) -> List[Tuple[Any, Any]]:
        """
        The function to compute the similarity of frames in the video suing OpenCV Histogram Comparison methods and
        then sort the frames based on the similarity score.

        Parameters
        -----------
        frame_directory: Dict[str, Any]
            The dictionary containing frame name as key and array image of frame as values.

        reference_image: str
            Pass a reference frame name (Starting frame of reconstructing video) to compute similarity by chaining
            one next to the other

        metric_choice: Histogram Comparison Methods [OpenCV]
            By Default: cv2.HISTCMP_BHATTACHARYYA

            OPENCV_METHOD_CHOICES
            ----------------------
                cv2.HISTCMP_CORREL - Computes Correlation
                cv2.HISTCMP_CHISQR - Computes Chi-Squared
                cv2.HISTCMP_INTERSECT - Computes Intersection
                cv2.HISTCMP_BHATTACHARYYA - Hellinger

        Returns
        --------
            The List of tuples sorted, where each tuple has frame name and corresponding
                similarity score with respect to the image reference chosen.

        """
        distance_metrics = {}
        for (idx, hist) in frame_directory.items():
            distance = cv2.compareHist(frame_directory[reference_image], hist, metric_choice)
            distance_metrics[idx] = distance
        distance_metrics = sorted([(val, idx) for (idx, val) in distance_metrics.items()])
        return distance_metrics

    @staticmethod
    def frame_sorting_scipy(frame_directory: Dict[str, Any], reference_image: str,
                            metric_choice=dist.chebyshev) -> List[Tuple[Any, Any]]:

        """
        The function to compute the similarity of frames in the video suing OpenCV Histogram Comparison
        methods and then sort the frames based on the similarity score.

        Parameters
        -----------
        frame_directory: Dict[str, Any]
            The dictionary containing frame name as key and array image of frame as values.

        reference_image: str
            Pass a reference frame name (Starting frame of reconstructing video) to compute similarity by chaining
            one next to the other.

        metric_choice: Vector Distance Methods [Scipy]
            By Default: dist.chebyshe

            SCIPY_METHOD_CHOICES
            ----------------------
            dist.euclidean - Computes Euclidean Distance
            dist.cityblock - Computes Manhattan Distance
            dist.chebyshev - Computes ChebyShev Distance

        Returns
        --------
        The List of tuples sorted, where each tuple has frame name and corresponding
            similarity score with respect to the image reference chosen.

        """

        distance_metrics = {}
        for (idx, hist) in frame_directory.items():
            distance = metric_choice(frame_directory[reference_image], hist)
            distance_metrics[idx] = distance
        distance_metrics = sorted([(val, idx) for (idx, val) in distance_metrics.items()])
        return distance_metrics

    @staticmethod
    def get_video_from_frames(sorted_frame_list: List[np.array], file_name: str) -> None:
        """
        The function to generate the reconstructed video file from the sorted frames.

        Parameters
        -----------
        sorted_frame_list: List[np.array]
            The list of array images of Sorted Frames.

        file_name: str
            The filename for saving reconstructed video.


        Returns
        --------
        The Reconstructed video will be saved to the notebooks' directory of the project.

        """

        height, width, layers = sorted_frame_list[0].shape
        size = (width, height)
        output = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

        for i in range(len(sorted_frame_list)):
            output.write(sorted_frame_list[i])
        output.release()
        print("Video Saved Successfully..!")


"""
    -----------
    Main Method 
    ------------
    
    @staticmethod
    def get_reconstructed_video(
                                images, index,
                                reference_image, file_name,
                                method="OPENCV",
                                metric=cv2.HISTCMP_CORREL,
                                hash_threshold=20):
    
        result = video.frame_sort_by_opencv(index, reference_image, metric)
        sorted_frames = get_sorted_frames(images, result)
        clean_frames = eliminate_outlier_frame(images, result, hash_threshold)
        get_vedio_from_frames(clean_frames, file_name)
        print("Vedio Saved Successfully..!! ")

"""