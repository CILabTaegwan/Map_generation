import cv2
import imageio
import numpy as np


class VideoRecorder(object):
    def __init__(self, height=1080, width=1920, camera_id=0, fps=4):
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self):
        self.frames = []

    def record(self, frame):
        frame = frame.astype(np.float32)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = frame * 255.
        frame = np.array(frame, dtype=np.uint8)
        self.frames.append(frame)

    def save(self, path):

        imageio.mimsave(path, self.frames, fps=self.fps)
