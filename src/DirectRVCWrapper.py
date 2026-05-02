import sys
import os
from pathlib import Path
import logging

import numpy as np

# w-okada/voice-changer を読み込むためのパスを追加
sys.path.insert(0, str(Path(__file__).parent / "vcclient" / "server"))

# fairseq.tasks.text_to_speech のログを無効化
# 使用されていないのにもかかわらず、tensorboardX がインストールされていない環境で INFO ログが出るため
logging.getLogger("fairseq.tasks.text_to_speech").disabled = True

from voice_changer.utils.VoiceChangerParams import VoiceChangerParams
from const import StaticSlot
from data.ModelSlot import RVCModelSlot, loadSlotInfo
from voice_changer.RVC.RVCr2 import RVCr2
from voice_changer.VoiceChangerV2 import VoiceChangerV2
from downloader.SampleDownloader import downloadInitialSamples
from downloader.WeightDownloader import downloadWeight

class DirectRVCWrapper:
    """
    RVC と VoiceChangerV2 を直接操作するためのラッパークラス
    """

    def __init__(
        self,
        index_path: str = None,
        model_dir: str = "model_dir",
        pretrain_dir: str = "pretrain",
    ):
        self.index_path = index_path
        self.model_dir = model_dir

        self.params = VoiceChangerParams(
            model_dir=model_dir,
            content_vec_500=os.path.join(pretrain_dir, "checkpoint_best_legacy_500.pt"),
            content_vec_500_onnx=os.path.join(pretrain_dir, "content_vec_500.onnx"),
            content_vec_500_onnx_on=True,
            hubert_base=os.path.join(pretrain_dir, "hubert_base.pt"),
            hubert_base_jp=os.path.join(pretrain_dir, "rinna_hubert_base_jp.pt"),
            hubert_soft=os.path.join(pretrain_dir, "hubert-soft-0d54a1f4.pt"),
            nsf_hifigan=os.path.join(pretrain_dir, "nsf_hifigan/model"),
            sample_mode="production",
            crepe_onnx_full=os.path.join(pretrain_dir, "crepe_onnx_full.onnx"),
            crepe_onnx_tiny=os.path.join(pretrain_dir, "crepe_onnx_tiny.onnx"),
            rmvpe=os.path.join(pretrain_dir, "rmvpe.pt"),
            rmvpe_onnx=os.path.join(pretrain_dir, "rmvpe.onnx"),
            whisper_tiny=os.path.join(pretrain_dir, "whisper_tiny.pt"),
        )

    def _load_model(self, model: int | StaticSlot) -> RVCModelSlot:
        """
        モデルスロットの情報を読み込みます。
        """
        from voice_changer.RVC.RVCModelSlotGenerator import RVCModelSlotGenerator

        slotInfo = loadSlotInfo(self.model_dir, model)

        if not isinstance(slotInfo, RVCModelSlot):
            raise ValueError(f"Slot {model} is not an RVC model or does not exist. (Got type {slotInfo.voiceChangerType})")

        if self.index_path:
            slotInfo.indexFile = self.index_path

        if slotInfo.isONNX:
            if not getattr(slotInfo, "modelType", "").startswith("onnx"):
                slotInfo = RVCModelSlotGenerator._setInfoByONNX(slotInfo.modelFile, slotInfo)
                if not getattr(slotInfo, "modelType", "").startswith("onnx"):
                    slotInfo.modelType = "onnxRVC"
        else:
            slotInfo = RVCModelSlotGenerator._setInfoByPytorch(slotInfo.modelFile, slotInfo)

        return slotInfo

    def download_initial_models(self):
        """
        サンプルモデルをダウンロードします。
        """
        downloadWeight(self.params)
        downloadInitialSamples(self.params.sample_mode, self.model_dir)

    def initialize(
        self,
        gpu_id: int,
        model: int | StaticSlot,
        input_sample_rate: int = 48000,
        output_sample_rate: int = 48000,
        f0_method: str = "rmvpe_onnx",
        f0_up_key: int = 0,
        silent_threshold: float = 0.00001,
        extra_convert_size: int = 1024 * 4,
        index_ratio: float = 0,
        protect: float = 0.5,
        rvc_quality: int = 0,
        silence_front: int = 1,
    ):
        """
        RVC と VoiceChangerV2 の初期化を行います。
        gpu_id: 使用するGPUのIDを指定します。CPUで動かす場合は -1 を指定してください。
        model: 使用するモデルのスロット番号を指定します。
        input_sample_rate: 入力音声のサンプリングレートを指定します。
        output_sample_rate: 出力音声のサンプリングレートを指定します。
        f0_method: f0 抽出に使用するモデルを指定します。(デフォルト: "rmvpe_onnx")
            指定可能な値: "dio", "harvest", "crepe", "crepe_tiny", "crepe_full", "rmvpe", "rmvpe_onnx", "fcpe"
            (指定可能な値は voice-changer/server/voice_changer/RVC/pitchExtractor/PitchExtractorManager.py を参照)
        f0_up_key: ピッチシフト量を半音単位で指定します。
        silent_threshold: 無音判定の閾値を指定します。
        extra_convert_size: 追加変換サイズを指定します。
        index_ratio: インデックス比率を指定します。
        protect: 保護設定を指定します。
        rvc_quality: RVC品質設定を指定します。
        silence_front: サイレンスフロント設定を指定します。(0: off, 1: on)
        """

        slot = self._load_model(model)
        slot.defaultTune = f0_up_key

        self.rvc = RVCr2(self.params, slot)
        self.rvc.update_settings("f0Detector", f0_method)
        self.rvc.update_settings("tran", f0_up_key)
        self.rvc.update_settings("gpu", gpu_id)
        self.rvc.update_settings("silentThreshold", silent_threshold)
        self.rvc.update_settings("extraConvertSize", extra_convert_size)
        self.rvc.update_settings("indexRatio", index_ratio)
        self.rvc.update_settings("protect", protect)
        self.rvc.update_settings("rvcQuality", rvc_quality)
        self.rvc.update_settings("silenceFront", silence_front)
        # GPU 設定を行うと self.rvc.initialize() は不要

        self.vc = VoiceChangerV2(self.params)
        self.vc.setModel(self.rvc)
        self.vc.setInputSampleRate(input_sample_rate)
        self.vc.setOutputSampleRate(output_sample_rate)

    def process_chunk(self, data: np.ndarray) -> np.ndarray | None:
        """
        音声チャンクを処理します。
        data: float32 ndarray, [-1.0, 1.0]
        """
        int16_input = (data * 32767.5).astype(np.int16)

        try:
            output_int16, perf = self.vc.on_request(int16_input)
        except Exception as e:
            print(f"Exception in process_chunk: {e}")
            import traceback
            traceback.print_exc()
            return None

        return output_int16.astype(np.float32) / 32768.0

    def close(self):
        """
        モデルのクリーンアップを行います。
        """
        self.rvc = None
        self.vc = None
