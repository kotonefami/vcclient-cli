from abc import ABC, abstractmethod
import numpy as np

class Output(ABC):
    @abstractmethod
    def write(self, chunk: np.ndarray) -> None:
        """指定された音声チャンクを出力します。"""
        pass

    @abstractmethod
    def close(self) -> None:
        """出力リソースを解放し、後処理を行います。"""
        pass
