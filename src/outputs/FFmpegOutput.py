import av
import numpy as np
from outputs.Output import Output

class FFmpegOutput(Output):
    def __init__(self, url: str, sample_rate: int = 44100, codec_name: str = "aac"):
        self.url = url
        self.sample_rate = sample_rate
        self.codec_name = codec_name
        self._container = None
        self._stream = None
        self._closed = False

    def _ensure_open(self):
        if self._container is not None:
            return
        self._container = av.open(self.url, mode="w", format="mpegts", options={"pkt_size": "1316"})
        self._stream = self._container.add_stream(self.codec_name, rate=self.sample_rate)
        self._stream.layout = "mono"
        self._stream.format = "fltp"

    def write(self, chunk: np.ndarray) -> None:
        if self._closed:
            return
        self._ensure_open()

        frame = av.AudioFrame.from_ndarray(
            chunk.reshape(1, -1).astype(np.float32),
            format="fltp",
            layout="mono"
        )
        frame.sample_rate = self.sample_rate
        frame.pts = None

        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        try:
            if self._stream is not None:
                for packet in self._stream.encode(None):
                    if self._container is not None:
                        self._container.mux(packet)
        finally:
            if self._container is not None:
                try:
                    self._container.close()
                except Exception:
                    pass
                self._container = None
