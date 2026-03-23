"""
core/video_handler.py
Responsibility: Load video from file, 
read frames, and manage playback loop.
"""

import cv2
from config import settings

class VideoHandler:
    """Manage the video source."""

    def __init__(self):
        self.capture = None
        self.is_playing = False
        self.fps = 0
        self.total_frames = 0
        self.current_frame = 0
        self.width = 0
        self.heigth = 0

    def open(self, path: str) -> bool:
        """
        Open a video file.
        Returns True if it opened successfully.
        """
        self.release()
        self.capture = cv2.VideoCapture(path)

        if not self.capture.isOpened():
            self.capture = None
            return False
        
        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or settings.DEFAULT_FPS
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.heigth = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame = 0
        self.is_playing = False

        return True
    
    def read_frame(self):
        """
        Read the next frame of the video.
        Return (True, frame) or (False, None) if there are no more frames.
        """
        if self.capture is None: 
            return False, None
        
        ret, frame = self.capture.read()

        if ret:
            self.current_frame = int(
                self.capture.get(cv2.CAP_PROP_POS_FRAMES)
            )
        return ret, frame
    
    def get_progress(self) -> float:
        """
        Return the video progress as a percentage.
        """
        if self.total_frames == 0:
            return 0.0
        return self.current_frame / self.total_frames

    def is_opened(self) -> bool:
        """
        Indicates if there is a video loaded and open.
        """
        return self.capture is not None and self.capture.isOpened()
    
    def release(self):
        """
        Release the resources of the current video.
        """
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0

    def get_info(self) -> dict:
        """
        Returns a dictionary with the properties of video.
        """
        return {
            "fps": self.fps,
            "total_frames" : self.total_frames,
            "width": self.width,
            "height": self.heigth,
            "duration_sec": (
                self.total_frames /self.fps if self.fps > 0 else 0
            )
        }

