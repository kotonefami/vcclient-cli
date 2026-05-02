from abc import ABC, abstractmethod
import numpy as np

class Input(ABC):
    @abstractmethod
    def chunks(self, chunk_size: int):
        """与えられたサイズの音声チャンクを yield で返します。"""
        pass

    @abstractmethod
    def sample_rate(self) -> int:
        """入力音声のサンプルレートを返します。"""
        pass

    @abstractmethod
    def close(self) -> None:
        """入力が保持するリソースを解放します。"""
        pass
