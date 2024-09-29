import pathlib
from os import PathLike

import pytest

from mediahash import video

TEST_DATA_FOLDER = pathlib.Path(__file__).parent / "data"


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        (
            TEST_DATA_FOLDER / "sample.mp4",
            video.VideoInfo(
                size=98599,
                width=480,
                height=270,
                length=3,
                bitrate=261311,
                format="mov,mp4,m4a,3gp,3g2,mj2",
            ),
        ),
    ],
)
def test_video_info(filename: PathLike, expected):
    with open(filename, "rb") as fileobj:
        info = video.analyze(fileobj)
    assert info == expected
