"""Fair-audio equalization regression — mu-law codec, channel equalization, provenance-leak removal (docs/BENCHMARK.md). DGX-free."""
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from miltl.nibble.audio_decode import mulaw_codec, equalize_channel
import scripts.compose_hard_kormmp as ch


class TestMulaw(unittest.TestCase):
    def test_roundtrip_shape_bounded(self):
        rng = np.random.RandomState(0)
        x = rng.randn(2000).astype(np.float32) * 0.3
        y = mulaw_codec(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertLessEqual(float(np.max(np.abs(y))), float(np.max(np.abs(x))) + 1e-5)

    def test_monotone_preserving(self):
        # mu-law is lossy but preserves monotonicity (sign and magnitude order) -> high correlation
        x = np.linspace(-1, 1, 500).astype(np.float32)
        y = mulaw_codec(x)
        self.assertGreater(float(np.corrcoef(x, y)[0, 1]), 0.99)

    def test_empty(self):
        self.assertEqual(len(mulaw_codec(np.array([], np.float32))), 0)

    def test_equalize_length(self):
        x = np.random.RandomState(1).randn(1600).astype(np.float32)
        y = equalize_channel(x, 16000, codec=True)
        self.assertEqual(len(y), len(x))
        self.assertTrue(np.all(np.isfinite(y)))


class TestFairPool(unittest.TestCase):
    def setUp(self):
        # _rank_cold decodes audio -> replace with identity (keeps input order) in tests
        self._orig = ch._rank_cold
        ch._rank_cold = lambda uris: list(uris)

    def tearDown(self):
        ch._rank_cold = self._orig

    def _cases(self):
        harm = [{"case_id": f"h{i}", "class": "harm", "audio_uri": f"data/raw/fss/posts/{i}.wav"}
                for i in range(6)]
        ben = [{"case_id": f"b{i}", "class": "benign", "audio_uri": f"data/raw/normal/emotion_tagged/{i}.wav"}
               for i in range(8)]
        return harm + ben

    def _synth(self):
        out = []
        for i in range(5):
            out.append({"case_id": f"sh{i}", "label": 1, "transcript": "지금 계좌 이체 하세요 " * 3,
                        "slice": "synth-hard-harm", "pair_prosody": "cold"})
        for i in range(5):
            out.append({"case_id": f"sb{i}", "label": 0, "transcript": "네 오늘 날씨 좋네요 " * 3,
                        "slice": "synth-hard-benign", "pair_prosody": "warm"})
        return out

    def test_fair_removes_provenance_leak(self):
        """fair_audio: all carriers come from the benign (normal) corpus -> 0 FSS assignments = provenance-label collinearity removed."""
        out = ch.pair_synth(self._synth(), self._cases(), seed=1, fair_audio=True)
        self.assertTrue(out)
        for r in out:
            self.assertIn("normal/emotion_tagged", r["audio_uri"],
                          f"fair 모드인데 비-benign 출처 배정: {r['audio_uri']}")
            self.assertNotIn("fss", r["audio_uri"])
            self.assertTrue(r["meta"]["audio_fair"])

    def test_default_keeps_fss_for_harm(self):
        """Default (non-fair): harm=FSS, benign=normal (leak present) — stated explicitly as the regression contrast."""
        out = ch.pair_synth(self._synth(), self._cases(), seed=1, fair_audio=False)
        harm = [r for r in out if r["label"] == 1]
        ben = [r for r in out if r["label"] == 0]
        self.assertTrue(all("fss" in r["audio_uri"] for r in harm))
        self.assertTrue(all("normal" in r["audio_uri"] for r in ben))

    def test_fair_cold_warm_disjoint_carriers(self):
        """cold (harm) and warm (benign) carriers come from different halves, preserving the prosody contrast."""
        out = ch.pair_synth(self._synth(), self._cases(), seed=1, fair_audio=True)
        cold_uris = {r["audio_uri"] for r in out if r["label"] == 1}
        warm_uris = {r["audio_uri"] for r in out if r["label"] == 0}
        self.assertTrue(cold_uris.isdisjoint(warm_uris), "cold·warm 반송파가 겹침(운율 대비 소실)")

    def test_compose_fair_no_fss_harm(self):
        """Non-synthetic slices (easy-harm, etc.) also get benign-corpus harm carriers under fair -> 0 residual FSS leak."""
        txt = "안녕하세요 " * 200                              # 200 words (passes the 150~360 length filter)
        harm = [{"case_id": f"h{i}", "class": "harm", "transcript": txt,
                 "audio_uri": f"data/raw/fss/posts/{i}.wav"} for i in range(4)]
        ben = [{"case_id": f"b{i}", "class": "benign", "transcript": txt,
                "audio_uri": f"data/raw/normal/emotion_tagged/{i}.wav"} for i in range(6)]
        out = ch.compose(harm + ben, per_slice=3, seed=1, asr_track="000",
                         mirror_dir="", fair_audio=True)
        harm_rows = [r for r in out if r["label"] == 1]
        self.assertTrue(harm_rows)
        for r in harm_rows:
            self.assertNotIn("fss", r["audio_uri"], f"fair 인데 harm 이 FSS 반송파: {r['audio_uri']}")
            self.assertIn("normal", r["audio_uri"])


if __name__ == "__main__":
    unittest.main()
