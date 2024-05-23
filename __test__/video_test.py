import tempfile
from typing import IO

import av
import numpy as np
import pytest

from mediahash import video


@pytest.fixture
def generated_video():
    duration = 4
    fps = 24
    total_frames = duration * fps
    with tempfile.SpooledTemporaryFile() as fileobj:
        with av.open(fileobj, mode="w", format="mp4") as container:
            stream = container.add_stream("mpeg4", rate=fps)
            stream.width = 480
            stream.height = 320
            stream.pix_fmt = "yuv420p"

            for frame_i in range(total_frames):

                img = np.empty((480, 320, 3))
                img[:, :, 0] = 0.5 + 0.5 * np.sin(
                    2 * np.pi * (0 / 3 + frame_i / total_frames)
                )
                img[:, :, 1] = 0.5 + 0.5 * np.sin(
                    2 * np.pi * (1 / 3 + frame_i / total_frames)
                )
                img[:, :, 2] = 0.5 + 0.5 * np.sin(
                    2 * np.pi * (2 / 3 + frame_i / total_frames)
                )

                img = np.round(255 * img).astype(np.uint8)
                img = np.clip(img, 0, 255)

                frame = av.VideoFrame.from_ndarray(img, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)

            # Flush stream
            for packet in stream.encode():
                container.mux(packet)
        fileobj.seek(0)
        yield fileobj


def test_video_info_is_correct(generated_video: IO):
    info = video.analyze(generated_video)
    assert info.length == 4
    assert info.width == 480
    assert info.height == 320
