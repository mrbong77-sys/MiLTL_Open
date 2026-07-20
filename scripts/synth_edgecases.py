#!/usr/bin/env python3
"""Synthetic edge-case text generation (docs/BENCHMARK.md) — lexically decorrelated hard cases.

Prosody-transfer (matched-pair) premise: this script generates **text only**; audio is paired
with real audio by the composer.
  synth-hard-harm  : phishing that **fully avoids** the scam lexicon (grooming, softened
                     authority pressure, indirect demands) -> pair_prosody=cold (real FSS)
  synth-hard-benign: benign calls that **deliberately include** finance/authority vocabulary
                     (bank/loan consultation, cooperative, warm) -> pair_prosody=warm (real benign)
Lexical density is controlled (hard-harm ~= 0, hard-benign high). Deterministic expansion via slot variation.
Scenario diversity (docs/BENCHMARK.md): 8 harm types (institution, loan, family, refund,
  parcel, acquaintance, investment-coaching, subsidy impersonation),
  8 benign types (bank, card, insurance, telecom, delivery, government office, hospital, securities).
  All harm scenarios avoid the scam lexicon (decorrelation is preserved).

  python scripts/synth_edgecases.py --n-per 40 --seed 42 --out artifacts/manifest/synth_edgecases.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from miltl.native.channel_teacher import _SCAM, _THREAT, _DIRECTIVE, _URGENCY

_BANNED = list(_SCAM) + list(_THREAT) + list(_DIRECTIVE) + list(_URGENCY)  # avoidance targets for hard-harm

# Neutral dialogue fillers (to pad length) — shared by both slices
_FILLER = [
    "네 여보세요", "아 예 반갑습니다", "예예 듣고 있어요", "그러시군요", "아 그렇죠",
    "잠시만요", "네 맞아요", "아 예 알겠어요", "그러셨어요", "음 그러네요",
    "예 계속 얘기하셔도 돼요", "아 네 그 부분은 이해했어요", "그렇게 하는 게 좋겠네요",
    "저도 그렇게 생각했어요", "예 한번 볼게요", "아 그런 일이 있으셨구나", "네 괜찮아요",
    "그럼요 당연하죠", "아 예 시간 괜찮아요", "천천히 얘기해 주셔도 돼요",
]

# hard-harm scenarios (avoid scam lexicon; softened pressure/demands) — scenario tags for qualitative analysis. Sample 4 lines per list.
_HARM_SCEN = {
    "기관사칭": [
        "공공기관 확인 부서에서 연락드렸는데요", "위에서 한번 확인해 보라고 내려온 건이 있어서요",
        "고객님 명단에 이상한 게 하나 올라와 있더라고요", "이거 그냥 두시면 나중에 곤란해지실 수 있어서요",
        "화면에 떠 있는 그 숫자만 저한테 읽어주시면 제가 처리해요", "걱정 마시고 저만 믿고 따라오시면 돼요",
        "시간 지나면 저도 손 못 쓰니까 얼른 하는 게 나아요", "다른 분들도 이렇게 해결하고 계세요"],
    "대출사칭": [
        "낮은 금리로 갈아탈 수 있는 좋은 건이 하나 나와서요", "고객님 조건이면 한도가 꽤 높게 잡히시더라고요",
        "심사는 제가 다 도와드릴 테니 걱정 안 하셔도 돼요", "화면에 나오는 번호 몇 자리만 확인해 주시면 진행돼요",
        "이 조건은 자리가 몇 개 안 남아서 서두르시는 게 좋아요", "다른 고객님들도 이렇게 많이 받으셨어요",
        "제가 끝까지 챙겨드릴 테니까 편하게 맡기세요"],
    "가족사칭": [
        "엄마 나야 폰이 고장나서 다른 번호로 연락해", "내가 급하게 처리할 게 있는데 좀 도와줄 수 있어",
        "지갑을 잃어버려서 내 걸로는 아무것도 못 하고 있어", "링크 하나 보낼 테니까 거기 뜨는 대로만 좀 해줘",
        "엄마한테만 부탁하는 거야 다른 사람한텐 말 안 했어", "좀 서둘러 줘야 될 것 같아 시간이 없어",
        "이거 해결되면 내가 다시 제대로 설명할게"],
    "환급사칭": [
        "고객님 돌려받으실 게 하나 남아 있어서 연락드렸어요", "환급 대상이신데 아직 안 찾아가셨더라고요",
        "제가 도와드리면 오래 안 걸리고 받으실 수 있어요", "화면에 뜨는 그 숫자만 불러주시면 금방 넣어드려요",
        "기한 지나면 못 받으시니까 아까워서 알려드리는 거예요", "복잡한 건 제가 다 해드릴 테니 걱정 마세요",
        "고객님만 조용히 진행하시면 문제없어요"],
    "택배사칭": [
        "고객님 앞으로 온 물건이 하나 반송될 뻔했더라고요", "세관 쪽에 걸려 있는 게 있어서 제가 확인 도와드려요",
        "화면에 뜨는 숫자 몇 자리만 저한테 알려주시면 풀려요", "이거 그냥 두시면 물건이 다시 돌아가 버려요",
        "다른 분들도 이렇게 간단히 찾으셨어요", "제가 안내해 드리는 대로만 따라오시면 금방 끝나요",
        "조금만 늦으면 저도 도와드리기 어려워져요"],
    "지인사칭": [
        "부장님 저예요 폰이 망가져서 다른 걸로 연락드려요", "잠깐 처리할 게 있는데 좀 도와주실 수 있을까요",
        "제 걸로는 도무지 아무것도 안 돼서 부탁드리는 거예요", "링크 하나 보낼 테니 거기 뜨는 대로만 좀 해주세요",
        "부장님한테만 부탁드리는 거예요 다른 분껜 얘기 안 했어요", "조금 서둘러 주셔야 될 것 같아요 여유가 없어서요",
        "이거 정리되면 제가 다시 제대로 인사드릴게요"],
    "투자리딩": [
        "좋은 자리 하나 들어와서 고객님께 먼저 안내드려요", "이번 건은 거의 확실하게 재미 보실 수 있어요",
        "다른 분들은 벌써 다 들어와 계세요", "화면에 보이는 숫자만 저랑 맞춰주시면 돼요",
        "복잡한 건 제가 다 챙겨드릴 테니 편하게 맡기세요", "자리가 얼마 안 남아서 서두르시는 게 좋아요",
        "고객님만 조용히 진행하시면 문제없어요"],
    "지원금사칭": [
        "나라에서 나오는 지원 대상이신데 아직 안 받으셨더라고요", "제가 도와드리면 오래 안 걸리고 나와요",
        "화면에 뜨는 그 번호만 불러주시면 처리돼요", "기한 넘기면 못 받으시니까 알려드리는 거예요",
        "서류 같은 복잡한 건 제가 다 해드려요", "다른 분들도 이렇게 많이 받아 가셨어요",
        "고객님 조건이면 금액이 꽤 크게 잡히시더라고요"],
}
# hard-benign scenarios (include finance/authority vocabulary; cooperative, warm, benign consultation).
_BEN_SCEN = {
    "은행": [
        "고객님 안녕하세요 은행 상담센터입니다", "계좌 이체 한도는 원하시는 대로 조정 가능하세요",
        "이체 수수료는 이 통장이면 면제되세요", "본인 인증만 한 번 하시면 바로 처리돼요",
        "편하실 때 천천히 알아보셔도 괜찮아요", "궁금하신 거 있으면 언제든 물어보세요"],
    "카드": [
        "안녕하세요 카드 상담 담당입니다", "연회비랑 수수료 안내부터 도와드릴게요",
        "본인 인증 절차만 마치시면 발급 진행돼요", "명의 확인은 간단하게 끝나세요",
        "급하게 결정 안 하셔도 되니까 편하게 보세요", "제일 유리한 혜택으로 안내해 드릴게요"],
    "보험": [
        "네 보험 상담 도와드리는 상담원입니다", "명의 확인만 해주시면 보험금 처리 도와드려요",
        "보증금 관련해서도 안내해 드릴 수 있어요", "수수료 없이 진행되는 상품이라 부담 없으세요",
        "필요하시면 자료도 메일로 보내드릴게요", "천천히 검토하시고 결정하셔도 돼요"],
    "통신": [
        "고객님 통신 요금 상담센터입니다", "명의 변경은 인증 한 번이면 처리돼요",
        "요금제는 계좌 이체로 등록해 두시면 편하세요", "수수료 부담 없는 요금제로 안내해 드릴게요",
        "제가 자세히 설명드릴 테니 편하게 들으세요", "결정은 편하실 때 하셔도 됩니다"],
    "배송": [
        "고객님 택배 배송 안내 도와드리는 상담원입니다", "재배송은 원하시는 날짜로 잡아드릴 수 있어요",
        "부재중이시면 경비실에 맡겨 드릴까요", "송장 번호는 문자로도 보내드릴게요",
        "급하게 안 하셔도 되니 편하실 때 확인하세요", "궁금하신 점 있으면 언제든 문의 주세요"],
    "관공서": [
        "네 주민센터 민원 상담입니다", "서류는 온라인으로도 발급 가능하세요",
        "방문 안 하셔도 처리되는 건이라 편하세요", "본인 인증만 한 번 하시면 조회돼요",
        "처리 기간은 보통 이삼일 정도 걸려요", "필요하시면 안내문 메일로 보내드릴게요"],
    "병원": [
        "건강검진 예약 도와드리는 상담원입니다", "결과는 등기로 보내드리니 안심하세요",
        "추가 검사는 원하실 때 잡으시면 돼요", "수수료 관련 안내도 함께 도와드릴게요",
        "예약 변경은 편하실 때 연락 주세요", "천천히 검토하시고 정하셔도 괜찮아요"],
    "증권": [
        "네 증권사 고객센터입니다", "계좌 개설은 비대면으로도 가능하세요",
        "수수료는 이번 이벤트로 면제되세요", "본인 인증 절차만 마치시면 진행돼요",
        "투자 상담은 예약제로 도와드려요", "급하게 결정 안 하셔도 되니 편하게 보세요"],
}


def _weave(rng, signal_lines, n_words=200):
    """Interleave signal lines with fillers to build a dialogue near the target word count."""
    out = []
    si = 0
    while len(" ".join(out).split()) < n_words:
        if si < len(signal_lines) and rng.random() < 0.5:
            out.append(signal_lines[si]); si += 1
        else:
            out.append(rng.choice(_FILLER))
    while si < len(signal_lines):
        out.insert(rng.randrange(len(out) + 1), signal_lines[si]); si += 1
    return " ".join(out)


def _gen_scen(rng, scen_dict):
    """Pick one scenario -> sample 4 signal lines -> dialogue. Returns (scenario, text)."""
    scen = rng.choice(list(scen_dict.keys()))
    pool = scen_dict[scen]
    lines = rng.sample(pool, min(4, len(pool)))
    return scen, _weave(rng, lines, rng.randint(160, 320))


def _density(t):
    nw = max(len(t.split()), 1)
    return sum(t.count(w) for w in _BANNED) / nw * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="artifacts/manifest/synth_edgecases.jsonl")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = []
    for i in range(args.n_per):
        scen, t = _gen_scen(rng, _HARM_SCEN)
        rows.append({"case_id": f"synth_hh_{i}", "class": "harm", "modality": "text",
                     "transcript": t, "slice": "synth-hard-harm", "pair_prosody": "cold", "scenario": scen,
                     "label": 1, "meta": {"density": round(_density(t), 2), "n_words": len(t.split()), "scenario": scen}})
    for i in range(args.n_per):
        scen, t = _gen_scen(rng, _BEN_SCEN)
        rows.append({"case_id": f"synth_hb_{i}", "class": "benign", "modality": "text",
                     "transcript": t, "slice": "synth-hard-benign", "pair_prosody": "warm", "scenario": scen,
                     "label": 0, "meta": {"density": round(_density(t), 2), "n_words": len(t.split()), "scenario": scen}})
    import numpy as np
    hh = [r["meta"]["density"] for r in rows if r["label"] == 1]
    hb = [r["meta"]["density"] for r in rows if r["label"] == 0]
    if args.selftest or not args.out:
        print(f"[synth] hard-harm lexical-density med={np.median(hh):.2f}(≈0 target) · hard-benign med={np.median(hb):.2f}(high target)")
        print(f"  sample hard-harm:  {rows[0]['transcript'][:90]}...")
        print(f"  sample hard-benign:{rows[args.n_per]['transcript'][:90]}...")
        assert np.median(hh) < 0.7, f"hard-harm lexicon leak: {np.median(hh)}"
        assert np.median(hb) > np.median(hh) + 1.0, "hard-benign lexical density insufficient"
        print("[synth] Synthetic-text lexicon decorrelation OK. Audio=composer prosody-transfer (cold=real FSS · warm=real benign).")
        if args.selftest:
            return 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"[synth] {len(rows)} cases → {args.out}  (hard-harm med={np.median(hh):.2f}, hard-benign med={np.median(hb):.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
