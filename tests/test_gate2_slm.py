"""Gate-2 SLM — nibble fusion summary (pure) + backbone contract and missing-modality isolation (docs/ARCHITECTURE.md, DGX-free)."""
import unittest

from adapters.baselines.gate2_slm import Gate2SLM, summarize_nibbles, summarize_channels, _bits


class TestSummarizeChannels(unittest.TestCase):
    """Channel-bottleneck diagnosis -> Gate-2 prompt summary (XM up front). DGX-free."""

    def test_xm_high_values(self):
        # Raw signals (values) only — never inject a conclusion (benign/phishing) (prevents echoing)
        s = summarize_channels({"XM": 0.15, "cold": 0.99, "warmth": 0.35, "T": 0.42,
                                "I": 0.74, "F": 0.20, "E": 0.70})
        self.assertIn("XM=0.15", s)
        self.assertIn("잠복위협I=0.74", s)
        self.assertIn("보조 지표", s)                        # states transcript-first explicitly

    def test_no_baked_verdict(self):
        # Even at low XM, do not inject conclusions like "normal conversation traits" (prevents echo misjudgment of low-XM phishing)
        s = summarize_channels({"XM": 0.03, "cold": 0.55, "warmth": 0.10, "T": 0.67})
        self.assertNotIn("정상 대화 특성", s)
        self.assertIn("전사에 사칭·이체유도 있으면 피싱", s)

    def test_missing_keys_default_zero(self):
        s = summarize_channels({})                          # defensive: missing keys -> 0.0
        self.assertIn("XM=0.00", s)


class TestXAIExplain(unittest.TestCase):
    """Deterministic XAI reason/action generation (edge user-report evidence). Pure, reproducible."""

    def test_harm_reasons_and_action(self):
        from miltl.native.explain import explain_decision
        e = explain_decision({"XM": 0.15, "cold": 0.99, "warmth": 0.35, "F": 0.3, "I": 0.7, "E": 0.7},
                             "harm", "검찰청입니다 안전계좌로 이체하세요", p1=0.82)
        self.assertIn("HIGH-RISK", e["verdict"])
        self.assertTrue(any("mismatch" in r for r in e["reasons"]))         # XM cross-modal reason
        self.assertTrue(any("phishing-typical" in r for r in e["reasons"]))  # transcript cue
        self.assertTrue(any("report" in a for a in e["action"]))            # linked action

    def test_benign_clean_no_action(self):
        from miltl.native.explain import explain_decision
        e = explain_decision({"XM": 0.02, "F": 0.1, "I": 0.5}, "benign", "네 감사합니다", p1=0.1)
        self.assertIn("SAFE", e["verdict"])
        self.assertIn("no action needed", e["action"][0])

    def test_benign_with_cue_escalates_caution(self):
        from miltl.native.explain import explain_decision
        e = explain_decision({"XM": 0.02}, "benign", "계좌 인증 부탁드려요", p1=0.3)
        self.assertIn("caution", e["verdict"])                          # escalate to the safe side when a cue fires


class TestCascadeBanding(unittest.TestCase):
    """Cascade band routing + P/R/F1 (SLM- and audio-independent, pure)."""

    def test_prf(self):
        from scripts.gate2_cascade import _prf
        m = _prf([1, 1, 0, 0], [1, 0, 0, 1])               # tp1 fn1 tn1 fp1
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (1, 1, 1))
        self.assertAlmostEqual(m["recall"], 0.5)
        self.assertAlmostEqual(m["precision"], 0.5)

    def test_band_thresholds(self):
        import numpy as np
        p1 = np.array([0.1, 0.5, 0.9])
        band = np.where(p1 <= 0.4, 0, np.where(p1 >= 0.75, 2, 1))
        self.assertEqual(list(band), [0, 1, 2])             # benign, escalate, harm

    def test_judge_channels_record_no_model(self):
        # Model not loaded (DGX-free): summary and prompt records are always produced (for reproduction)
        g = Gate2SLM()
        jd = g.judge_channels("계좌가 정지되었습니다", {"XM": 0.15, "cold": 0.99, "warmth": 0.35})
        self.assertIn("summary", jd); self.assertIn("prompt", jd)
        self.assertIn("XM=0.15", jd["summary"])
        self.assertIn("판정", jd["prompt"])                  # includes the verdict instruction (reproduction prompt)
        self.assertIsNone(jd["decision"])                   # no verdict without a model

    def test_ledger_write(self):
        import json, tempfile, os
        from scripts.gate2_cascade import _write_records
        led = [{"call_id": "c1", "slice": "synth-hard-harm", "label": 1,
                "gate1": {"p1": 0.6, "band": "escalate", "XM": 0.15},
                "gate2": {"summary": "XM=0.15 ...", "prompt": "...", "rationale": "따뜻한 말+냉정 음성",
                          "raw": "...판정: 예", "decision": "harm"},
                "final": "harm", "outcome": "TP"}]
        m = {"recall": 1.0, "precision": 1.0, "f1": 1.0, "fp": 0}
        d = tempfile.mkdtemp(); lp = os.path.join(d, "led.jsonl")
        _write_records(led, m, m, 1, 1, "oracle", 0.4, 0.75, lp, "", False)
        r = json.loads(open(lp, encoding="utf-8").read().strip())
        self.assertEqual(r["gate2"]["decision"], "harm")    # LMM verdict preserved
        self.assertIn("rationale", r["gate2"])              # rationale preserved (post-hoc analysis)


class TestSummarize(unittest.TestCase):
    def test_bits_layout(self):
        self.assertEqual(_bits(0b1010), (1, 0, 1, 0))      # T I F E
        self.assertEqual(_bits(15), (1, 1, 1, 1))

    def test_occupancy_and_ramp(self):
        # First half F=0 (value 8 = T only), second half F=1 (value 10 = T+F) -> F occupancy 0.5, F run 3, harm slope +1.0
        s = summarize_nibbles([8, 8, 8, 10, 10, 10], [], p1=0.62)
        self.assertIn("텍스트:", s)
        self.assertIn("F0.50", s)
        self.assertIn("F지속max 3/6", s)
        self.assertIn("harm기울기 +1.00", s)
        self.assertIn("Gate-1 p1 0.62", s)

    def test_fe_cooccurrence(self):
        # Value 11 = T+F+E -> F∧E co-occurrence 1.0
        self.assertIn("F∧E공기 1.00", summarize_nibbles([11, 11], []))

    def test_missing_modality(self):
        s = summarize_nibbles([8, 10], [None, None], state="TEXT-ONLY")
        self.assertIn("파형: (결측)", s)
        self.assertIn("가용성 TEXT-ONLY", s)


class TestContract(unittest.TestCase):
    def test_default_backbone(self):
        d = Gate2SLM()
        self.assertEqual(d.model_name, "Qwen/Qwen2.5-0.5B-Instruct")  # PoC first choice
        self.assertFalse(d.finetune)
        self.assertIn("Qwen2.5-0.5B", d.name)

    def test_ablation_backbone_swap(self):
        d = Gate2SLM(model_name="LiquidAI/LFM2.5-230M")               # ablation swap
        self.assertEqual(d.model_name, "LiquidAI/LFM2.5-230M")

    def test_backend_hint_without_torch(self):
        try:
            import torch  # noqa: F401
            self.skipTest("torch 설치됨")
        except ImportError:
            pass
        with self.assertRaises(RuntimeError) as ctx:
            Gate2SLM()._ensure()
        self.assertIn("torch", str(ctx.exception))

    def test_score_before_fit_is_zero(self):
        self.assertEqual(Gate2SLM().score("x", "요약"), 0.0)          # no model -> 0


if __name__ == "__main__":
    unittest.main()
