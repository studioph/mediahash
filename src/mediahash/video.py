from dataclasses import dataclass
import hashlib
import logging
from typing import IO, Iterable, Protocol
from typing_extensions import Buffer

import av
import numpy as np
import pywt
from av.error import InvalidDataError  # pylint: disable=no-name-in-module
from scipy.spatial import distance
from av.stream import Stream

from .util import reraise

from .types import MediaError, MediaFingerprint

LOG = logging.getLogger(__name__)

FRAME_THRESHOLD = 1000
HASH_SIZE = 8
MILLIS_PER_SECOND = 1_000_000
CHUNK_SIZE = 1024 * 64
FP_DECIMAL_PLACES = 4

extensions = (
    ".mp4",
    ".mkv",
    ".avi",
    ".wmv",
    ".mov",
    ".m4v",
    ".webm",
    ".mpg",
    ".ts",
    ".mpeg",
    ".mpg",
)

rethrow = reraise(InvalidDataError, as_=MediaError)

class HashObj(Protocol):
    def update(self, __data: Buffer):
        ...

@dataclass(frozen=True, kw_only=True)
class VideoInfo:
    width: int
    height: int
    size: int
    length: int
    bitrate: int

class VideoFingerprint:
    def __init__(self, vhash: np.ndarray):
        self._fp = vhash

    def __sub__(self, other: np.ndarray) -> float:
        return round(distance.cosine(self._fp, other), FP_DECIMAL_PLACES)

    def __bytes__(self) -> bytes:
        return self._fp.tobytes()

@staticmethod
@rethrow
def analyze(media: IO) -> VideoInfo:
    LOG.debug("Analyzing %s", media.name)
    with av.open(media, mode="r") as container:
        vidstream = container.streams.video[0]
        duration = container.duration // MILLIS_PER_SECOND
        return VideoInfo(
            size=container.size,
            length=duration,
            bitrate=container.bit_rate,
            width=vidstream.width,
            height=vidstream.height,
        )
    
def _estimate_stream_size(stream: Stream) -> int:
    if not stream.codec_context.bit_rate:
        return 0
    seconds = float(stream.duration * stream.time_base) if stream.duration else stream.container.duration / 1_000_000
    size_bits = stream.codec_context.bit_rate * seconds
    return round(size_bits / 8)


def _read_from_packets(packets: Iterable, amount: int, digest: HashObj):
    """Reads a fixed amount of data from media stream packets"""
    read = 0
    while read < amount:
        remaining = amount - read
        packet = next(packets)
        # Read the whole packet if less than remaining, otherwise read remaining bytes
        to_read = min(remaining, packet.size)
        digest.update(bytes(packet)[:to_read])
        read += to_read
    assert read == amount

@staticmethod
@rethrow
def mhash(media: IO) -> bytes:
    with av.open(media) as container:
        chksm = hashlib.md5()
        # Set up packet generator to ignore empty packets
        packets = (p for p in container.demux(video=0) if p.pos)
        _read_from_packets(packets, CHUNK_SIZE, chksm)
        container.seek(container.duration // 2)
        _read_from_packets(packets, CHUNK_SIZE, chksm)
    return chksm.digest()

@staticmethod
@rethrow
def checksum(media: IO) -> bytes:
    with av.open(media) as container:
        chksm = hashlib.md5()
        # Set up packet generator to ignore empty packets
        packets = (p for p in container.demux() if p.pos)
        for packet in packets:
            chksm.update(packet)
    return chksm.digest()

@staticmethod
@rethrow
def fingerprint(media: IO) -> MediaFingerprint:
    LOG.debug("Fingerprinting %s", media.name)
    with av.open(media, mode="r") as container:
        instream = container.streams.video[0]
        if instream.frames > FRAME_THRESHOLD:
            LOG.debug("Number of frames exceeds threshold, using only keyframes")
            instream.codec_context.skip_frame = "NONKEY"
        scaled_size = HASH_SIZE * HASH_SIZE

        graph = av.filter.Graph()
        _link_nodes(
            graph.add_buffer(template=instream),
            graph.add("scale", f"{scaled_size}:{scaled_size}"),
            graph.add("setsar", "1:1"),
            graph.add("format", "gray"),
            graph.add("buffersink"),
        )
        graph.configure()

        hashes = np.zeros((HASH_SIZE, HASH_SIZE), dtype=np.float32)
        for inframe in container.decode(instream):
            graph.push(inframe)
            outframe = graph.pull()
            hash_vector = _whash(outframe.to_ndarray(), hash_size=HASH_SIZE)
            hashes += hash_vector
    return VideoFingerprint(hashes)


def _link_nodes(*nodes):
    for prev_node, next_node in zip(nodes, nodes[1:]):
        prev_node.link_to(next_node)


def _whash(
    frame, hash_size, image_scale=None, mode="haar", remove_max_haar_ll=True
) -> np.ndarray:
    """Calculates the wavlet hash of a video frame. Based on https://github.com/JohannesBuchner/imagehash whash"""

    if image_scale is not None:
        assert image_scale & (image_scale - 1) == 0, "image_scale is not power of 2"
    else:
        image_natural_scale = 2 ** int(np.log2(min(frame.shape)))
        image_scale = max(image_natural_scale, hash_size)

    ll_max_level = int(np.log2(image_scale))

    level = int(np.log2(hash_size))
    assert hash_size & (hash_size - 1) == 0, "hash_size is not power of 2"
    assert level <= ll_max_level, "hash_size in a wrong range"
    dwt_level = ll_max_level - level

    pixels = np.divide(frame, 255.0, dtype=np.float32)

    # Remove low level frequency LL(max_ll) if @remove_max_haar_ll using haar filter
    if remove_max_haar_ll:
        coeffs = pywt.wavedec2(pixels, "haar", level=ll_max_level)
        coeffs = list(coeffs)
        coeffs[0] *= 0
        pixels = pywt.waverec2(coeffs, "haar")

    # Use LL(K) as freq, where K is log2(@hash_size)
    coeffs = pywt.wavedec2(pixels, mode, level=dwt_level)
    dwt_low = coeffs[0]
    return dwt_low


@staticmethod
def parse_fingerprint(__bytes: bytes) -> MediaFingerprint:
    arr = np.frombuffer(__bytes, dtype=np.float32)
    return VideoFingerprint(arr)
