import ffmpeg
import numpy

from src import get_video_info


class VideoFile:
    """
    A class to help facilitate reading data from video files using ffmpeg-python
    """
    def __init__(self, filename):
        self.filename = filename
        # Extract framerate from ffprobe info
        info = ffmpeg.probe(filename)
        self.width, self.height, self.framerate = get_video_info(info)

    def __len__(self) -> int:
        """
        Returns length in number of frames
        :return: number of frames
        """

    def __getitem__(self, item) -> numpy.array:
        pass

