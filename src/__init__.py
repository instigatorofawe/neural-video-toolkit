from typing import Tuple


def get_video_info(info) -> Tuple[int, int, str]:
    """
    Get frame width, height, and framerate of a video file
    :param info: Dict containing results of ffmpeg.probe()
    :return: Tuple containing width, height, and framerate
    """
    framerate = None
    width = None
    height = None

    streams = info['streams']

    for i in range(0, len(streams)):
        if 'width' in streams[i].keys():
            width = streams[i]['width']
        if 'height' in streams[i].keys():
            height = streams[i]['height']
        if streams[i]['codec_type'] == 'video' and 'r_frame_rate' in streams[i].keys():
            framerate = streams[i]['r_frame_rate']

    return int(width), int(height), framerate
