"""Corpus adapters — convert each source's transcripts into a common Call (utterance list + label + meta).

Supported:
  - fss_calls          : FSS phishing meta.json → dialogue turns (scammer/victim)      label=phishing
  - dailydialog_calls  : AI-Hub daily-dialogue-130 json → sessionInfo.dialog utterances label=benign
  - ksponspeech_calls  : KsponSpeech .trn transcripts → bundles of 40 utterances        label=benign
  - aihub_dialogue_calls: AI-Hub multi-session/goal/topic dialogue json (incl. inside zip, arbitrary schema) label=benign
The common output Call flows through tiler → featurizer → schema.CallStream.
"""
from __future__ import annotations

import io
import json
import os
import re
import wave
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

# ASR track selection (uniform-evaluation setting (b), see docs/BENCHMARK.md): when set, prefer the
# audio's sibling ASR transcript (<stem><track>) as the transcript source.
# E.g. ASR_TRACK=.asr_light.txt (main eval) or .asr_fw.txt (reverse ablation). Unset = legacy behavior (backward compatible).
_ASR_TRACK = os.environ.get("ASR_TRACK", "")

_TURN_RE = re.compile(r"^\s*(사기범|피해자|상담원|고객|직원|안내|남자|여자)\s*[:：]\s*(.*)$")
_UTT_KEYS = ("utterance", "text", "sentence", "selectedContents", "content", "contents")


@dataclass
class Call:
    call_id: str
    source: str
    label: Optional[str]              # "phishing" | "benign" | None
    utterances: List[str] = field(default_factory=list)
    split_keys: dict = field(default_factory=dict)
    audio_path: Optional[str] = None  # audio file for the waveform path (mp3/mp4/wav/pcm). None = text-only.
    audio_paths: Optional[List[str]] = None  # multiple audio files (decode then concatenate) — short-utterance pcm into 2 minutes.


# ------------------------------------------------------------------ FSS phishing
def _fss_turns(text: str) -> List[str]:
    out = []
    for line in (text or "").splitlines():
        m = _TURN_RE.match(line)
        if m and m.group(2).strip():
            out.append(m.group(2).strip())
    return out


_AUDIO_EXT = (".mp3", ".mp4", ".m4a", ".wav", ".pcm", ".aac", ".ogg", ".flac", ".webm", ".wma")


def fss_audio_calls(root: str = "data/raw/fss") -> Iterator[Call]:
    """FSS phishing audio (posts under voice/voice2/video) → calls that have audio. **For the waveform channel** (P2).

    Each post = 1 call, with audio_path set. If a transcript (turns) exists it is included (fusion);
    otherwise audio-only (waveform only). voice2 (audio, no transcript) and video (mp4) are also
    included — rglob covers all posts subtrees."""
    for meta in sorted(Path(root).rglob("posts/*/meta.json")):
        try:
            rec = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            continue
        pdir = meta.parent
        aud = next((str(p) for e in _AUDIO_EXT for p in sorted(pdir.glob(f"*{e}"))), None)
        if aud is None:
            continue
        nid = str(rec.get("nttId", pdir.name))
        # Uniform-evaluation setting (b): when ASR_TRACK is set, use the audio's ASR transcript
        # (instead of the board text). Unset = meta text turns.
        asr = _asr_sidecar(Path(aud))
        utts = [asr] if asr else _fss_turns(rec.get("text", ""))
        yield Call(call_id=f"fssaud_{nid}", source="fss_audio", label="phishing",
                   utterances=utts,
                   split_keys={"source": "fss_audio", "speaker": nid}, audio_path=aud)


def fss_calls(root: str = "data/raw/fss/posts", min_turns: int = 4) -> Iterator[Call]:
    """FSS meta.json → conversational calls only (≥min_turns turns). All phishing.

    Root robustness: if `<root>/*/meta.json` is empty, fall back to `<root>/posts/*/meta.json`
    (datasets.yaml uses root=data/raw/fss while the actual meta files live under posts/ — avoids
    path mismatch)."""
    base = Path(root)
    metas = sorted(base.glob("*/meta.json")) or sorted((base / "posts").glob("*/meta.json"))
    for meta in metas:
        try:
            rec = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            continue
        turns = _fss_turns(rec.get("text", ""))
        if len(turns) < min_turns:
            continue
        nid = rec.get("nttId", meta.parent.name)
        yield Call(call_id=f"fss_{nid}", source="fss", label="phishing",
                   utterances=turns, split_keys={"source": "fss", "channel": "fss_grnom", "speaker": nid})


# ------------------------------------------------------- daily-dialogue-130 benign
def _pick_text(u) -> str:
    if isinstance(u, str):
        return u
    if not isinstance(u, dict):
        return ""
    for k in _UTT_KEYS:
        v = u.get(k)
        if isinstance(v, str) and v.strip():
            return v
    ctx = u.get("context")
    if isinstance(ctx, dict) and isinstance(ctx.get("contents"), str):
        return ctx["contents"]
    cands = [v for v in u.values() if isinstance(v, str)]
    return max(cands, key=len) if cands else ""


# AI-Hub free-conversation speech (general adults) schema: 발화정보.stt (noise tags like (NO:)/(SN:))
_STT_NOISE = re.compile(r"\([A-Z]{1,4}:\)")


def _stt_text(obj) -> str:
    """발화정보.stt (or top-level stt) → transcript with noise tags removed and whitespace normalized. '' if absent."""
    if not isinstance(obj, dict):
        return ""
    info = obj.get("발화정보")
    s = info.get("stt") if isinstance(info, dict) else None
    if not isinstance(s, str):
        s = obj.get("stt")
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", _STT_NOISE.sub(" ", s)).strip()


def _json_transcript(obj) -> str:
    """Arbitrary dialogue json → transcript text. Order: utterance list (_utts_from_obj) → stt → generic (_pick_text)."""
    us = _utts_from_obj(obj)
    if us:
        return " ".join(us)
    s = _stt_text(obj)
    if s:
        return s
    t = _pick_text(obj)
    return _clean_kspon(t) if t else ""


def _dialog_utts(obj: dict) -> List[str]:
    utts: List[str] = []
    sinfo = obj.get("sessionInfo")
    sessions = sinfo if isinstance(sinfo, list) else [sinfo] if isinstance(sinfo, dict) else []
    for s in sessions:
        if isinstance(s, dict) and isinstance(s.get("dialog"), list):
            for u in s["dialog"]:
                t = _pick_text(u)
                if t:
                    utts.append(t)
    return utts


def dailydialog_calls(root: str, min_utts: int = 4) -> Iterator[Call]:
    """daily-dialogue-130 label json (recursive over directory) → session utterances. All benign.

    DGX: if zip-compressed, extract first, or point --root at a small extracted subset. (Bulk zip
    handling is a separate follow-up option.)"""
    for jp in sorted(Path(root).rglob("*.json")):
        try:
            obj = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        utts = _dialog_utts(obj)
        if len(utts) < min_utts:
            continue
        mid = ""
        try:
            mid = obj.get("multisessionInfo", {}).get("multisessionID", "")
        except Exception:
            pass
        yield Call(call_id=f"daily_{jp.stem}", source="dailydialog130", label="benign",
                   utterances=utts, split_keys={"source": "dailydialog130", "speaker": mid or jp.stem})


# ------------------------------------------------- KsponSpeech benign (speech transcripts)
# KsponSpeech transcript special-notation cleanup:
#   (orthography)/(pronunciation) → keep orthography | strip b/ n/ l/ o/ u/ noise tags | strip + * / special chars
_KSPON_DUAL = re.compile(r"\(([^()/]*)\)/\([^()]*\)")
_KSPON_NOISE = re.compile(r"(?:(?<=\s)|^)[bnlou]/")
_KSPON_SPECIAL = re.compile(r"[+*/]")


def _read_text(p: Path) -> str:
    b = p.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", "ignore")


def _clean_kspon(t: str) -> str:
    t = _KSPON_DUAL.sub(r"\1", t)
    t = _KSPON_NOISE.sub(" ", t)
    t = _KSPON_SPECIAL.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def _kspon_utts(root: str) -> List[str]:
    """Prefer aggregated .trn ('path :: transcript'); fall back to per-utterance .txt. Returns cleaned transcripts."""
    utts: List[str] = []
    for trn in sorted(Path(root).rglob("*.trn")):
        for line in _read_text(trn).splitlines():
            tx = line.split("::", 1)[1] if "::" in line else line
            c = _clean_kspon(tx)
            if c:
                utts.append(c)
    if not utts:
        for txt in sorted(Path(root).rglob("*.txt")):
            c = _clean_kspon(_read_text(txt))
            if c:
                utts.append(c)
    return utts


def ksponspeech_audio_calls(root: str, secs_per_call: float = 120.0, sr: int = 16000,
                            bits: int = 16, channels: int = 1) -> Iterator[Call]:
    """KsponSpeech pcm (≈2.7s per utterance) → 2-minute benign pseudo-calls (waveform channel, P2). Concatenates pcm.

    Each call = secs_per_call worth of consecutive pcm files (audio_paths). build_streams --with-wave
    decodes and concatenates them for prosody → waveform-nibble. Headerless raw pcm duration is
    estimated as size ÷ (sr·bytes·ch)."""
    bps = sr * (bits // 8) * channels
    group, dur, gid = [], 0.0, 0
    for p in sorted(Path(root).rglob("*.pcm")):
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        group.append(str(p)); dur += sz / bps
        if dur >= secs_per_call:
            yield Call(call_id=f"ksponaud_{gid:05d}", source="ksponspeech_audio", label="benign",
                       utterances=[], split_keys={"source": "ksponspeech_audio", "speaker": f"grp{gid}"},
                       audio_paths=group)
            gid += 1; group, dur = [], 0.0
    if group and dur >= secs_per_call * 0.5:
        yield Call(call_id=f"ksponaud_{gid:05d}", source="ksponspeech_audio", label="benign",
                   utterances=[], split_keys={"source": "ksponspeech_audio", "speaker": f"grp{gid}"},
                   audio_paths=group)


def _find_trn_files(root: Path, pcm_dirs: set) -> List[Path]:
    """.trn under root; if none, walk up ancestors (≤4) and search (KsponSpeech keeps _scripts in a separate subtree).
    Prefer .trn whose stem matches a pcm parent-folder name (eval_clean.trn↔eval_clean/) — avoids the huge train.trn."""
    found = list(root.rglob("*.trn"))
    anc = root
    for _ in range(4):
        if found:
            break
        anc = anc.parent
        if anc == anc.parent:
            break
        found = list(anc.rglob("*.trn"))
    if pcm_dirs:
        rel = [t for t in found if t.stem in pcm_dirs]   # eval_clean.trn stem=eval_clean
        if rel:
            found = rel
    seen, uniq = set(), []      # dedup duplicate script copies (by stem)
    for t in sorted(found):
        if t.stem not in seen:
            seen.add(t.stem); uniq.append(t)
    return uniq


def _kspon_trn_map(root: str, pcm_dirs: Optional[set] = None) -> dict:
    """Aggregated .trn ('path :: transcript') → {pcm basename/stem: cleaned transcript}. For waveform-text pairing.
    .trn is searched under root or in the ancestor _scripts tree (entries matching pcm parent-folder names preferred)."""
    m = {}
    for trn in _find_trn_files(Path(root), pcm_dirs or set()):
        for line in _read_text(trn).splitlines():
            if "::" not in line:
                continue
            path, tx = line.split("::", 1)
            c = _clean_kspon(tx)
            if not c:
                continue
            name = Path(path.strip()).name          # KsponSpeech_000001.pcm
            m[name] = c
            m[Path(name).stem] = c                  # also key without extension
    return m


def ksponspeech_dual_calls(root: str, secs_per_call: float = 120.0, sr: int = 16000,
                           bits: int = 16, channels: int = 1) -> Iterator[Call]:
    """KsponSpeech .trn (gold transcripts) + .pcm (audio) → **dual-modal benign calls** (no ASR needed).

    Each call = secs_per_call worth of consecutive pcm files (audio_paths) + their transcripts (utterances).
    build_streams --with-wave produces both text→text_tife (PEINN) & audio→wave_tife (prosody) → bimodal
    data for fusion training (C). Pcm files without a matched transcript have missing text (that call is
    waveform-only) — late fusion tolerates missing modalities."""
    pcms = sorted(Path(root).rglob("*.pcm"))
    pcm_dirs = {p.parent.name for p in pcms}         # {eval_clean, eval_other} — for selecting relevant .trn
    tmap = _kspon_trn_map(root, pcm_dirs)
    bps = sr * (bits // 8) * channels
    group, utts, dur, gid, matched = [], [], 0.0, 0, 0
    for p in pcms:
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        tx = tmap.get(p.name) or tmap.get(p.stem)
        if not tx:                                   # no aggregated .trn → fall back to sibling .txt next to pcm
            sib = p.with_suffix(".txt")
            if sib.exists():
                tx = _clean_kspon(_read_text(sib))
        group.append(str(p)); utts.append(tx or ""); dur += sz / bps
        if tx:
            matched += 1
        if dur >= secs_per_call:
            yield Call(call_id=f"kspondual_{gid:05d}", source="ksponspeech_dual", label="benign",
                       utterances=[u for u in utts if u],   # matched transcripts only (drop empty strings)
                       split_keys={"source": "ksponspeech_dual", "speaker": f"grp{gid}"},
                       audio_paths=group)
            gid += 1; group, utts, dur = [], [], 0.0
    if group and dur >= secs_per_call * 0.5:
        yield Call(call_id=f"kspondual_{gid:05d}", source="ksponspeech_dual", label="benign",
                   utterances=[u for u in utts if u],
                   split_keys={"source": "ksponspeech_dual", "speaker": f"grp{gid}"},
                   audio_paths=group)


# ------------------------------------------- generic audio+transcript dual (e.g. Sample_voice wav+json)
def _audio_dur_s(p: Path, sr: int, bits: int, channels: int) -> Optional[float]:
    """Audio duration in seconds. wav = measured from header, pcm = size estimate, other (compressed) = unknown (None)."""
    ext = p.suffix.lower()
    if ext in (".pcm", ".raw"):
        try:
            return p.stat().st_size / (sr * (bits // 8) * channels)
        except OSError:
            return None
    if ext == ".wav":
        try:
            with wave.open(str(p), "rb") as w:
                return w.getnframes() / (w.getframerate() or sr)
        except Exception:
            return None
    return None


def _asr_sidecar(p: Path) -> str:
    """When ASR_TRACK is set, prefer reading the audio's sibling ASR transcript (<stem><track>, produced by asr_transcribe). '' if absent."""
    if not _ASR_TRACK:
        return ""
    try:
        s = p.with_suffix(_ASR_TRACK)          # same naming as asr_transcribe (with_suffix)
    except ValueError:
        return ""
    return (_read_text(s).strip() if s.exists() else "")


def _sibling_transcript(p: Path, tmap: dict) -> str:
    """Transcript for an audio file: (ASR transcript first if ASR_TRACK is set) → sibling .json → sibling .txt → .trn map."""
    a = _asr_sidecar(p)
    if a:
        return a
    j = p.with_suffix(".json")
    if j.exists():
        obj = _load_json_bytes(j.read_bytes())
        tx = _json_transcript(obj) if obj is not None else ""
        if tx:
            return tx
    t = p.with_suffix(".txt")
    if t.exists():
        c = _clean_kspon(_read_text(t))
        if c:
            return c
    return tmap.get(p.name) or tmap.get(p.stem) or ""


def _zip_dual_items(zpath: Path, sr: int, bits: int, channels: int):
    """Audio members inside a zip + transcripts (same-stem json/txt) → [(uri, transcript, dur_s)]. No extraction (member read)."""
    items = []
    try:
        zf = zipfile.ZipFile(zpath)
    except zipfile.BadZipFile:
        return items
    infos = [zi for zi in zf.infolist() if not zi.is_dir()]
    tmap = {}                                            # stem → (member, ext)
    for zi in infos:
        e = Path(zi.filename).suffix.lower()
        if e in (".json", ".txt"):
            tmap.setdefault(Path(zi.filename).stem, (zi.filename, e))
    for zi in sorted(infos, key=lambda z: z.filename):
        e = Path(zi.filename).suffix.lower()
        if e not in _AUDIO_EXT:
            continue
        tx = ""
        tm = tmap.get(Path(zi.filename).stem)
        if tm:
            tn, te = tm
            try:
                b = zf.read(tn)
            except (zipfile.BadZipFile, RuntimeError):
                b = b""
            if te == ".json":
                obj = _load_json_bytes(b)
                tx = _json_transcript(obj) if obj is not None else ""
            else:
                tx = _clean_kspon(b.decode("utf-8", "ignore"))
        if e == ".wav":                                  # exact duration from header
            try:
                import io as _io
                with wave.open(_io.BytesIO(zf.read(zi.filename)), "rb") as w:
                    dur = w.getnframes() / (w.getframerate() or sr)
            except Exception:
                dur = zi.file_size / (sr * (bits // 8) * channels)
        elif e in (".pcm", ".raw"):
            dur = zi.file_size / (sr * (bits // 8) * channels)
        else:
            dur = None
        items.append((f"zip:{zpath}!{zi.filename}", tx, dur))
    return items


def audio_dual_calls(root: str, source: str = "audio_dual", secs_per_call: float = 120.0,
                     sr: int = 16000, bits: int = 16, channels: int = 1) -> Iterator[Call]:
    """Arbitrary audio (wav/pcm/mp3…, on disk or **inside zip**) + transcript (sibling json/txt/.trn) → dual-modal benign.

    Handles KsponSpeech (pcm+trn), Sample_voice (wav+json inside zip), etc. through a single path. Zips
    stay compressed (member read, `zip:<zip>!<member>` URI). build_streams --with-wave yields
    text_tife (transcript→PEINN) + wave_tife (audio→prosody)."""
    root_p = Path(root)
    auds = sorted(q for e in _AUDIO_EXT for q in root_p.rglob(f"*{e}"))
    pcm_dirs = {q.parent.name for q in auds}
    has_pcm = any(q.suffix.lower() in (".pcm", ".raw") for q in auds)
    tmap = _kspon_trn_map(root, pcm_dirs) if has_pcm else {}
    # unified (uri, transcript, dur) stream: disk files + zip members
    items = [(str(p), _sibling_transcript(p, tmap), _audio_dur_s(p, sr, bits, channels)) for p in auds]
    for zp in sorted(root_p.rglob("*.zip")):
        items.extend(_zip_dual_items(zp, sr, bits, channels))

    group, utts, dur, gid = [], [], 0.0, 0

    def _mk(g, u, i):
        return Call(call_id=f"{source}_{i:05d}", source=source, label="benign",
                    utterances=[x for x in u if x],
                    split_keys={"source": source, "speaker": f"grp{i}"}, audio_paths=g)

    for uri, tx, d in items:
        group.append(uri); utts.append(tx or "")
        dur += d if d else 8.0
        if dur >= secs_per_call:
            yield _mk(group, utts, gid)
            gid += 1; group, utts, dur = [], [], 0.0
    if group and dur >= secs_per_call * 0.5:
        yield _mk(group, utts, gid)


def sample_voice_calls(root: str) -> Iterator[Call]:
    """Sample_voice (wav+json inside zip) → dual-modal benign. Wrapper over audio_dual_calls with source set."""
    yield from audio_dual_calls(root, source="sample_voice")


def freetalk_text_calls(root: str, utts_per_call: int = 40, min_utts: int = 8) -> Iterator[Call]:
    """Free-conversation speech (general adults) [label] zip → **text benign** (no audio, per-utterance stt).

    Each json = 1 utterance (발화정보.stt). **Group stt per session (member parent folder)** into calls of
    utts_per_call utterances (≈2 min). Spontaneous conversational style → diversifies the text
    distribution. Audio is not in these [label] zips (raw audio distributed separately)."""
    def _emit(sess, buf, gid):
        return Call(call_id=f"freetalk_{gid:06d}", source="freetalk", label="benign",
                    utterances=list(buf), split_keys={"source": "freetalk", "speaker": sess})

    cur, buf, gid = None, [], 0
    for zp in sorted(Path(root).rglob("*.zip")):
        try:
            zf = zipfile.ZipFile(zp)
        except zipfile.BadZipFile:
            continue
        for zi in sorted(zf.infolist(), key=lambda z: z.filename):
            if zi.is_dir() or not zi.filename.lower().endswith(".json"):
                continue
            sess = str(Path(zi.filename).parent)
            if sess != cur:                          # session boundary → flush remainder (if ≥min_utts)
                if len(buf) >= min_utts:
                    yield _emit(cur, buf, gid); gid += 1
                cur, buf = sess, []
            try:
                obj = _load_json_bytes(zf.read(zi.filename))
            except (zipfile.BadZipFile, RuntimeError):
                continue
            s = _stt_text(obj) if obj else ""
            if s:
                buf.append(s)
            if len(buf) >= utts_per_call:
                yield _emit(sess, buf, gid); gid += 1; buf = []
    if len(buf) >= min_utts:
        yield _emit(cur, buf, gid)


# ------------------------------------------- emotion-tagged free dialogue (raw wav + label json parallel trees)
_EMO_FINE = ("기쁨", "화남", "놀라움", "두려움", "슬픔", "사랑스러움")  # SpeakerEmotionTarget


def _wav_index(root: str) -> dict:
    """wav under root → {stem: path}. Loose wav uses the real path; wav inside zip uses a `zip:<zip>!<member>` URI
    (audio_decode.decode_to_pcm decodes without extraction). Also covers the full corpus (TS raw zips)."""
    idx: dict = {}
    for p in sorted(Path(root).rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext == ".wav":
            idx.setdefault(p.stem, str(p))                       # loose first (real path = decode guaranteed)
        elif ext == ".zip":
            try:
                zf = zipfile.ZipFile(p)
            except zipfile.BadZipFile:
                continue
            for zi in zf.infolist():
                if zi.is_dir() or not zi.filename.lower().endswith(".wav"):
                    continue
                idx.setdefault(Path(zi.filename).stem, f"zip:{p}!{zi.filename}")
    return idx


def emotion_dialog_calls(root: str, source: str = "emotion_dialog",
                         require_wav: bool = True) -> Iterator[Call]:
    """AI-Hub emotion-tagged free dialogue → dual-modal benign calls (with emotion labels; hard-case material).

    Layout: 01.원천데이터/**/<name>.wav (48kHz stereo)  +  02.라벨링데이터/**/<name>.json (transcript+emotion).
    json.Conversation[*].Text becomes utterances; the most frequent SpeakerEmotionTarget (excluding
    none/neutral) becomes the call emotion. split_keys records emotion (dominant fine emotion),
    anger_ratio (fraction of angry utterances) and category (dominant category) → for hard-case
    selection. wav goes into audio_path — build_streams --with-wave resamples → prosody → wave_tife.

    **Full corpus supported**: wav/json inside zips are read without extraction (_wav_index, _iter_json_bytes).
    _Sample (loose) and full (TS/TL zip) layouts share the same code path. Meta-jsons without a
    Conversation are naturally skipped.
    require_wav=True (default): labels without a wav pair are skipped (dual only). False also emits text-only."""
    wavs = _wav_index(root)                                # stem → real path | zip:URI (loose+zip)
    gid = 0
    seen: set = set()
    for stem, b in _iter_json_bytes(str(root)):           # label json: loose + zip members
        if stem in seen:                                  # avoid duplicate stems across loose/zip
            continue
        obj = _load_json_bytes(b)
        if not isinstance(obj, dict):
            continue
        conv = obj.get("Conversation") or []
        # Collect utterance text **together with** timing (StartTime/EndTime), order preserved.
        # For precise native mel-text alignment.
        utts, times = [], []
        for u in conv:
            if not (isinstance(u, dict) and u.get("Text", "").strip()):
                continue
            utts.append(u["Text"].strip())
            try:
                times.append([float(u.get("StartTime", 0.0)), float(u.get("EndTime", 0.0))])
            except (TypeError, ValueError):
                times.append([0.0, 0.0])
        if len(utts) < 4:                                 # naturally excludes transcript-less meta-json (Wav/File/Noise)
            continue
        wp = wavs.get(stem)
        if require_wav and wp is None:
            continue
        seen.add(stem)
        fine = [u.get("SpeakerEmotionTarget") for u in conv if isinstance(u, dict)]
        fine = [e for e in fine if e in _EMO_FINE]
        cats = [u.get("SpeakerEmotionCategory") for u in conv if isinstance(u, dict)]
        cats = [c for c in cats if c]
        n_utt = max(1, len([u for u in conv if isinstance(u, dict)]))
        anger = sum(1 for e in fine if e == "화남") / n_utt
        dom = max(set(fine), key=fine.count) if fine else "중립"
        domcat = max(set(cats), key=cats.count) if cats else "중립"
        has_t = any(e > 0 for _, e in times)              # only attach when timings are valid
        yield Call(call_id=f"{source}_{gid:05d}", source=source, label="benign",
                   utterances=utts, audio_path=wp,
                   split_keys={"source": source, "speaker": stem,
                               "emotion": dom, "category": domcat,
                               "anger_ratio": round(anger, 3),
                               **({"utt_times": times} if has_t else {})})
        gid += 1


def ksponspeech_calls(root: str, utts_per_call: int = 40) -> Iterator[Call]:
    """KsponSpeech transcripts → benign pseudo-calls (bundles of utts_per_call utterances ≈2 min). All benign.

    ⚠️ KsponSpeech is utterance-level (≈2.7s), so consecutive utterances are bundled for the 2-minute
    window. The actual .trn format is validated with samples on the DGX (path separators/encoding/
    special notation)."""
    utts = _kspon_utts(root)
    for i in range(0, len(utts), utts_per_call):
        grp = utts[i:i + utts_per_call]
        if len(grp) < 4:
            continue
        gid = i // utts_per_call
        yield Call(call_id=f"kspon_{gid:05d}", source="ksponspeech", label="benign",
                   utterances=grp, split_keys={"source": "ksponspeech", "speaker": f"grp{gid}"})


# ------------------------------------------- AI-Hub generic dialogue (multi-session/goal/topic) benign
def _find_utt_list(obj, maxdepth: int = 6):
    """Recursively search arbitrary JSON for an utterance list (list of dicts with text keys) → return best match."""
    best = [None, 0]

    def visit(o, d):
        if d > maxdepth:
            return
        if isinstance(o, list):
            sc = sum(1 for x in o if isinstance(x, dict) and any(k in x for k in _UTT_KEYS))
            if sc > best[1]:
                best[0], best[1] = o, sc
            for x in o[:100]:
                visit(x, d + 1)
        elif isinstance(o, dict):
            for v in o.values():
                visit(v, d + 1)

    visit(obj, 0)
    return best[0] if best[1] else None


def _utts_from_obj(obj) -> List[str]:
    """Extract utterance text: sessionInfo→dialog first, else recursive fallback for arbitrary schemas."""
    us = _dialog_utts(obj) if isinstance(obj, dict) else []
    if us:
        return us
    lst = _find_utt_list(obj)
    return [t for u in (lst or []) for t in (_pick_text(u),) if t]


def _load_json_bytes(b: bytes):
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return json.loads(b.decode(enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return None


def _iter_json_bytes(root: str) -> Iterator[tuple]:
    """.json under root (disk + inside zip) → (stem, bytes). Zip members read without extraction."""
    for p in sorted(Path(root).rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext == ".json":
            yield (p.stem, p.read_bytes())
        elif ext == ".zip":
            try:
                zf = zipfile.ZipFile(p)
            except zipfile.BadZipFile:
                continue
            for zi in zf.infolist():
                if zi.is_dir() or not zi.filename.lower().endswith(".json"):
                    continue
                try:
                    yield (Path(zi.filename).stem, zf.read(zi.filename))
                except (zipfile.BadZipFile, RuntimeError):
                    continue


def aihub_dialogue_calls(root: str, min_utts: int = 8) -> Iterator[Call]:
    """AI-Hub Korean multi-session/goal/topic/daily dialogue json → benign pseudo-calls (1 session file = 1 call).

    Json inside zips is read without extraction. sessionInfo→dialog first; arbitrary schemas use the
    recursive fallback. ⚠️ These corpora are mostly **text chat** (not speech) → for the text-nibble
    (PEINN) path. Speaking rate here is based on message timestamps, which differs from speech
    utterance rate — real speech rate is measured with KsponSpeech (audio)."""
    for sid, b in _iter_json_bytes(root):
        obj = _load_json_bytes(b)
        if obj is None:
            continue
        utts = _utts_from_obj(obj)
        if len(utts) < min_utts:
            continue
        yield Call(call_id=f"aihub_{sid}", source="aihub", label="benign",
                   utterances=utts, split_keys={"source": "aihub", "speaker": sid})


# ------------------------------------------- AI-Hub tsv dialogue (empathetic/persona) benign
_TSV_UTT_RE = re.compile(r"(?i)(utterance_text|발화문|발화|문장|sentence|\btext\b|content|script)")
_TSV_EMO_RE = re.compile(r"(?i)(emotion|감정|정서|기분)")
# Non-utterance columns (false-positive blocking): id/type/date/no etc. Excludes utterance_id, utterance_type, regDate, etc.
_TSV_BLOCK = ("_id", "type", "date", "time", "seq", "terminate", "flag", "url", "path", "count")
# AI-Hub 046 empathetic emotion labels — not inside the tsv; encoded in zip/folder names (TS_기쁨_…).
_EMOTIONS = ("기쁨", "당황", "분노", "불안", "상처", "슬픔")


def _decode_bytes(b: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", "ignore")


def _emotion_from(label: str) -> str:
    for e in _EMOTIONS:
        if e in label:
            return e
    return ""


def _pick_utt_col(header: List[str], rows: List[List[str]]) -> Optional[int]:
    """Pick the utterance column — exclude id/type/date headers, prefer text hints + **max mean cell length** (free text).

    Key to avoiding utterance_id/utterance_type false positives: the length criterion never picks
    code columns like '1'/'0'."""
    n = len(header)
    cand = [j for j, h in enumerate(header)
            if h.lower() not in ("id", "no") and not any(b in h.lower() for b in _TSV_BLOCK)]
    if not cand:
        cand = list(range(n))
    hinted = [j for j in cand if _TSV_UTT_RE.search(header[j])]
    pool = hinted or cand

    def avglen(j):
        vals = [r[j] for r in rows if j < len(r)]
        return sum(len(v) for v in vals) / len(vals) if vals else 0.0

    return max(pool, key=avglen) if pool else None


_DISK = "__disk__"


def _iter_delim_bytes(root: str, exts=(".tsv", ".csv")) -> Iterator[tuple]:
    """tsv/csv under root (disk + inside zip) → (stem, ext, bytes, arc).

    **Per-emotion round-robin**: group by emotion (기쁨/분노/… in the file/zip name), then iterate one
    emotion at a time → a limited sample (--limit) covers all 6 emotions evenly (no bias even when a
    single zip mixes several emotions)."""
    from collections import OrderedDict
    members = []          # (stem, ext, key, ref, arc)  key=_DISK or zip path
    for p in sorted(Path(root).rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in exts:
            members.append((p.stem, ext, _DISK, str(p), str(p)))
        elif ext == ".zip":
            try:
                zf = zipfile.ZipFile(p)
                for zi in zf.infolist():
                    e = Path(zi.filename).suffix.lower()
                    if not zi.is_dir() and e in exts:
                        members.append((Path(zi.filename).stem, e, str(p), zi.filename,
                                        p.name + "/" + zi.filename))
                zf.close()
            except zipfile.BadZipFile:
                continue

    # diversity key = emotion (file name first, else archive name) → round-robin per emotion when present.
    groups: "OrderedDict[str, list]" = OrderedDict()
    for m in members:
        stem, _ext, key, _ref, arc = m
        gk = _emotion_from(stem) or _emotion_from(arc) or key
        groups.setdefault(gk, []).append(m)
    glist = list(groups.values())

    zcache: dict = {}

    def _read(key: str, ref: str) -> bytes:
        if key == _DISK:
            return Path(ref).read_bytes()
        zf = zcache.get(key)
        if zf is None:
            zf = zcache[key] = zipfile.ZipFile(key)
        return zf.read(ref)

    idx = [0] * len(glist)
    remaining = len(members)
    while remaining > 0:
        for gi, mem in enumerate(glist):
            if idx[gi] < len(mem):
                stem, ext, key, ref, arc = mem[idx[gi]]
                idx[gi] += 1
                remaining -= 1
                try:
                    yield (stem, ext, _read(key, ref), arc)
                except (zipfile.BadZipFile, RuntimeError, OSError):
                    continue


def callcenter_calls(root: str, min_chars: int = 100) -> Iterator[Call]:
    """Call-center hard negatives (business support calls — accounts/authentication/identity checks,
    **lexically similar to phishing**) → benign.
    A **hard distribution** contrasting with KorMMP diverse (emotion dialogue), for validating content
    detector discrimination.

    Zip auto-detection: many per-utterance .txt = KtelSpeech (per-session-folder .txt transcripts +
    .wav audio → **dual**; wavs as multiple zip:URIs = the session's utterances, build_streams decodes
    then concatenates). json = civil-complaint corpus (turn list → text)."""
    root_p = Path(root)
    zips = [root_p] if root_p.suffix.lower() == ".zip" else list(root_p.rglob("*.zip"))
    for zp in zips:
        try:
            zf = zipfile.ZipFile(zp)
        except (zipfile.BadZipFile, FileNotFoundError):
            continue
        names = zf.namelist()
        n_txt = sum(1 for n in names if n.lower().endswith(".txt"))
        n_json = sum(1 for n in names if n.lower().endswith(".json"))
        if n_txt >= n_json:                               # KtelSpeech — txt+wav per session folder
            from collections import defaultdict
            txts, wavs = defaultdict(list), defaultdict(list)
            for zi in zf.infolist():
                if zi.is_dir():
                    continue
                low = zi.filename.lower()
                parent = zi.filename.rsplit("/", 1)[0]
                if low.endswith(".txt"):
                    txts[parent].append(zi.filename)
                elif low.endswith(".wav"):
                    wavs[parent].append(zi.filename)
            for parent in sorted(txts):
                parts = [_decode_bytes(zf.read(f)).strip() for f in sorted(txts[parent])]
                text = " ".join(p for p in parts if p).strip()
                if len(text) < min_chars:
                    continue
                aps = [f"zip:{zp}!{w}" for w in sorted(wavs.get(parent, []))]
                sess = parent.rsplit("/", 1)[-1]
                yield Call(call_id=f"callcenter_ktel:{sess}", source="callcenter_ktel",
                           label="benign", utterances=[text],
                           audio_path=(aps[0] if aps else None), audio_paths=(aps or None),
                           split_keys={"source": "callcenter_ktel", "domain": "callcenter",
                                       "topic": "상담"})
        else:                                             # civil complaints — json turn list → text-only
            member = next((zi.filename for zi in zf.infolist()
                           if zi.filename.lower().endswith(".json") and not zi.is_dir()), None)
            if not member:
                continue
            try:
                turns = json.loads(_decode_bytes(zf.read(member)))
            except json.JSONDecodeError:
                continue
            if not isinstance(turns, list):
                continue
            from collections import defaultdict
            calls = defaultdict(list)
            for t in turns:
                if not isinstance(t, dict):
                    continue
                did = str(t.get("대화셋일련번호", ""))
                txt = " ".join(str(t.get(k, "")).strip()
                               for k in ("고객질문(요청)", "상담사답변", "질문", "답변")
                               if str(t.get(k, "")).strip())
                if did and txt:
                    try:
                        sn = int(t.get("문장번호", 0))
                    except (ValueError, TypeError):
                        sn = 0
                    calls[did].append((sn, txt))
            for did, turns_ in calls.items():
                text = " ".join(t for _, t in sorted(turns_)).strip()
                if len(text) < min_chars:
                    continue
                yield Call(call_id=f"callcenter_minwon:{did}", source="callcenter_minwon",
                           label="benign", utterances=[text],
                           split_keys={"source": "callcenter_minwon", "domain": "callcenter",
                                       "topic": "민원"})


def tsv_dialogue_calls(root: str, min_utts: int = 8) -> Iterator[Call]:
    """AI-Hub tsv dialogue (046 empathetic, 044 persona) → benign pseudo-calls. 1 file = 1 call.

    The utterance column is robustly detected by **max mean length** (free text) — avoids
    utterance_id/type false positives. Emotion is not inside the tsv but extracted from the
    **zip/file path** (TS_기쁨_…) → tagged in split_keys (for anger/anxiety false-alarm validation)."""
    from collections import Counter
    for sid, ext, b, arc in _iter_delim_bytes(root):
        text = _decode_bytes(b)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        delim = "\t" if ext == ".tsv" else ","
        header = lines[0].split(delim)
        rows = [ln.split(delim) for ln in lines[1:]]
        ui = _pick_utt_col(header, rows[:30])
        if ui is None:
            continue
        ei = next((i for i, h in enumerate(header) if _TSV_EMO_RE.search(h)), None)
        utts, emos = [], []
        for cells in rows:
            if ui < len(cells) and cells[ui].strip():
                utts.append(cells[ui].strip())
                if ei is not None and ei < len(cells) and cells[ei].strip():
                    emos.append(cells[ei].strip())
        if len(utts) < min_utts:
            continue
        emo = (Counter(emos).most_common(1)[0][0] if emos else "") or _emotion_from(arc)
        yield Call(call_id=f"tsv_{sid}", source="aihub_tsv", label="benign", utterances=utts,
                   split_keys={"source": "aihub_tsv", "speaker": sid, "emotion": emo})


ADAPTERS = {"fss": fss_calls, "fss_audio": fss_audio_calls, "dailydialog130": dailydialog_calls,
            "ksponspeech": ksponspeech_calls, "ksponspeech_audio": ksponspeech_audio_calls,
            "ksponspeech_dual": ksponspeech_dual_calls,
            "audio_dual": audio_dual_calls, "sample_voice": sample_voice_calls,
            "emotion_dialog": emotion_dialog_calls, "freetalk": freetalk_text_calls,
            "aihub": aihub_dialogue_calls, "aihub_tsv": tsv_dialogue_calls,
            "callcenter": callcenter_calls}
