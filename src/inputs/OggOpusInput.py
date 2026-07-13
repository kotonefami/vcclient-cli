from inputs.FFmpegInput import FFmpegInput

class OggOpusInput(FFmpegInput):
    """低遅延 UDP/Opus 通信のための薄いラッパークラス。

    FFmpegInput を継承し、Ogg/Opus の入力に特化したオプションを注入する。
    """

    def __init__(self, port: int = 20012, **kwargs):
        url = f"udp://0.0.0.0:{port}?overrun_nonfatal=1&fifo_size=50000"

        input_args = [
            "-f", "ogg",
            "-c:a", "libopus",
            "-analyzeduration", "0",
            "-probesize", "32",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-flush_packets", "1",
        ]

        super().__init__(url=url, input_args=input_args, **kwargs)
