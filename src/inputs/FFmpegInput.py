"""FFmpeg を使った汎用入力クラスです。

ProcessInput を継承し、FFmpeg に特化したコマンド構築と
デフォルトオプションを提供します。
"""

import logging
from inputs.ProcessInput import ProcessInput

logger = logging.getLogger("app")


class FFmpegInput(ProcessInput):
    """FFmpeg を使って音声入力を受け取るクラス。

    input_args で入力側のオプションを外部から注入できます。
    指定しない場合は低遅延向けのデフォルト値が使用されます。
    """

    DEFAULT_INPUT_ARGS: list[str] = [
        "-analyzeduration", "0",
        "-probesize", "32",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-flush_packets", "1",
    ]

    def __init__(
        self,
        url: str,
        sample_rate: int,
        chunk_size: int = 1024,
        input_args: list[str] | None = None,
    ):
        """FFmpeg 入力を作成します。

        :param url: FFmpeg の -i に渡す入力 URL
        :param sample_rate: 出力サンプルレート
        :param chunk_size: チャンクサイズ
        :param input_args: FFmpeg の入力オプション（-i の前に追加）
        """
        self.url = url
        self._input_args = input_args if input_args is not None else self.DEFAULT_INPUT_ARGS

        cmd = [
            "ffmpeg",
            *self._input_args,
            "-i", self.url,
            "-f", "f32le",
            "-ac", "1",
            "-ar", str(sample_rate),
            "pipe:1",
        ]
        super().__init__(cmd, sample_rate, chunk_size)
