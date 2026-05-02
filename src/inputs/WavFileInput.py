import numpy as np
from inputs.Input import Input

class WavFileInput(Input):
    def __init__(self, file_path: str):
        self.file_path = file_path
        import scipy.io.wavfile as wavfile
        self.sr, self.data = wavfile.read(file_path)

        # float32 [-1.0, 1.0] へ正規化
        if self.data.dtype == np.int16:
            self.data = self.data.astype(np.float32) / 32768.0
        elif self.data.dtype == np.int32:
            self.data = self.data.astype(np.float32) / 2147483648.0
        elif self.data.dtype == np.float64:
            self.data = self.data.astype(np.float32)

        # モノラルへ変換
        if self.data.ndim > 1:
            self.data = np.mean(self.data, axis=1)

        self._closed = False

    def sample_rate(self) -> int:
        return self.sr

    def chunks(self, chunk_size: int):
        if self._closed:
            return
        for i in range(0, len(self.data), chunk_size):
            chunk = self.data[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            yield chunk

    def close(self) -> None:
        self._closed = True
