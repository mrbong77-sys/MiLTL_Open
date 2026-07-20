"""Nibble pattern-recognition core (see docs/ARCHITECTURE.md).

Each 8-second segment of a call is one observation ("nibble"). Channel signals are
thresholded into compact per-segment codes, and the accumulated nibble sequence is
classified benign/harm by Gate-1; only uncertain-band calls escalate to the Gate-2 SLM.

This public package exports the modules required by the canonical benchmark pipeline;
exploratory/legacy modules of the private research repo are intentionally not shipped.
"""
from .tife import TIFE, TIFEProvider, MockTIFEProvider
from .encoder import NibbleThresholds, NibbleEncoder, NibbleAccumulator, unpack, T_BIT, I_BIT, F_BIT, E_BIT
from .features import NibbleFeatures, extract_features
from .gate1 import Gate1Scorer, RuleLogisticGate1
from .synth import SynthCall, synth_call, synth_dataset, synth_mm_dataset
from .gate1_train import LogisticRegression
from .recalibrate import (
    calibrate_thresholds, combined_call_auc, re_encode_stream, ChannelReport,
    train_combined_gate, score_call, fused_call_auc, pair_benign_modalities,
)
from .seq_adaptor import (
    MultiScaleCNNAdaptor, FeatureContext, sequence_matrix, encode_segment,
    train_cnn_adaptor, ENCODINGS, ADAPTORS,
)
from .dataset import MMWindow, load_calls_jsonl, windows_from_calls, LABEL_MAP
from .schema import (
    SCHEMA_VERSION, SegmentRecord, CallStream, fuse_byte, split_byte, build_call_stream, attach_wave,
)
from .tiler import tile_by_words, time_tile
from .featurize import TextFeaturizer, MockTextFeaturizer, PeinnTextFeaturizer
from .corpora import (
    Call, fss_calls, fss_audio_calls, dailydialog_calls, ksponspeech_calls,
    ksponspeech_audio_calls, ksponspeech_dual_calls, audio_dual_calls, sample_voice_calls,
    emotion_dialog_calls, freetalk_text_calls, aihub_dialogue_calls, tsv_dialogue_calls, callcenter_calls, ADAPTERS,
)
