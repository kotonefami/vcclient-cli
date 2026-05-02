import numpy as np
import scipy.io.wavfile as wavfile
from outputs.Output import Output

class WavFileOutput(Output):
    def __init__(self, file_path: str, sample_rate: int):
        self.file_path = file_path
        self.sample_rate = sample_rate
        self.output_chunks = []

    def write(self, chunk: np.ndarray) -> None:
        self.output_chunks.append(chunk)

    def close(self) -> None:
        if self.output_chunks:
            final_out = np.concatenate(self.output_chunks)
            final_out_int16 = (final_out * 32767).astype(np.int16)
            wavfile.write(self.file_path, self.sample_rate, final_out_int16)
