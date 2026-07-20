#!/usr/bin/env python3
"""
fetch_fss_voicephishing.py — collector of real voice-phishing recordings from the FSS (for local execution on the DGX Spark).

Target: Financial Supervisory Service (FSS) voice-phishing awareness boards (multiple sources, selected via --source)
      voice  = B0000206/200690 (audio, previously collected)
      voice2 = B0000207/200691 (audio, same format, 23 pages)
      video  = B0000203/200686 (mp4 video, 13 pages)
      An arbitrary board can be specified directly with --board B0000xxx --menu-no xxxxxx.
      After collection, --summary writes a git-safe summary (raw audio/body text excluded) to artifacts/fss_sources/<label>.json.

Design notes (docs/DATA_ACCESS.md):
  * Run this collector **locally on the DGX Spark, which can reach FSS**, not in a remote CI container.
    (From remote containers, FSS blocks access with 403 / network policy.)
  * FSS is an eGovFrame (Korean e-government standard framework) BBS. It uses the
    list.do/view.do/file-download pattern, but selectors/parameters may change after a site
    redesign, so we parse **defensively**:
      - view links: broadly detect anchors carrying an nttId
      - audio: <audio>/<source> + .mp3/.wav/.m4a attachment links + getFile/FileDown/download-style endpoints
  * Recommended workflow: first run `--dry-run` to see what gets picked up, then fine-tune only
    the parts that need it against the actual HTML structure.

Output:
  data/raw/fss/
    index.jsonl                 one line of metadata per post
    posts/<nttId>/meta.json     post metadata + body text (text modality)
    posts/<nttId>/<file>.mp3    audio (PII — never committed to git)

Usage:
  pip install requests beautifulsoup4 tqdm
  python scripts/fetch_fss_voicephishing.py --pages 1-5 --dry-run
  python scripts/fetch_fss_voicephishing.py --pages 1-20 --out data/raw/fss --resume
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urljoin, urlparse, parse_qs, unquote

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - environment hint
    sys.exit("Missing dependencies: pip install requests beautifulsoup4 tqdm")

try:
    from tqdm import tqdm
except ImportError:  # fall back to a no-op if tqdm is missing
    def tqdm(x, **k):  # type: ignore
        return x

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fss")

BASE = "https://www.fss.or.kr"


@dataclass(frozen=True)
class Board:
    """One FSS eGovFrame board — board id + menuNo + human-readable label / default output path."""
    board: str            # e.g. "B0000206"
    menu_no: str          # e.g. "200690"
    label: str            # e.g. "voice"
    media: str = "audio"  # "audio" | "video" (mp4 boards)

    @property
    def list_url(self) -> str:
        return f"{BASE}/fss/bbs/{self.board}/list.do"

    @property
    def view_url(self) -> str:
        return f"{BASE}/fss/bbs/{self.board}/view.do"

    def default_out(self) -> str:
        # voice (legacy) keeps data/raw/fss for backward compatibility; additional sources go under data/raw/fss/<label>.
        return "data/raw/fss" if self.label == "voice" else f"data/raw/fss/{self.label}"


# Known sources (all share the same eGovFrame structure — list.do/view.do/getFile.do).
SOURCES = {
    "voice":  Board("B0000206", "200690", "voice", "audio"),   # existing: "The Scammer's Voice" (audio), already collected
    "voice2": Board("B0000207", "200691", "voice2", "audio"),  # additional: same format (audio), 23 pages
    "video":  Board("B0000203", "200686", "video", "video"),   # additional: mp4 video, 13 pages
}
DEFAULT_BOARD = SOURCES["voice"]
# Backward-compat aliases (referenced by existing code and the selftest).
VIEW_URL = DEFAULT_BOARD.view_url
LIST_URL = DEFAULT_BOARD.list_url
MENU_NO = DEFAULT_BOARD.menu_no

# Impersonate a regular browser (the FSS WAF blocks non-browser UAs with 403)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
}

# FSS "The Scammer's Voice" serves audio as mp4 (video container)/mp3 etc., so include video extensions too.
AUDIO_EXT = (".mp3", ".wav", ".m4a", ".ogg", ".aac", ".wma", ".mp4", ".webm", ".opus", ".flac")
DOWNLOAD_HINTS = (
    "getfile", "filedown", "download", "atchfile", "media", "stream",
    "movie", "/cmm/fms/", "/comm/getfile",
)

# Regex that scans the raw HTML directly for media URLs / download endpoints.
# (Covers JS-injected URLs, data-* attributes, and strings inside <script> that a BeautifulSoup anchor-walk misses.)
_EXT_RE = "|".join(e.lstrip(".") for e in AUDIO_EXT)
RAW_MEDIA_RE = re.compile(
    r"""(?xi)
    (?:https?:)?//[^\s"'<>()]+?\.(?:%s)(?:\?[^\s"'<>()]*)?   # 절대 URL .ext
    | /[^\s"'<>()]+?\.(?:%s)(?:\?[^\s"'<>()]*)?               # 루트상대 .ext
    | [^\s"'<>()]*(?:getFile|FileDown|download|streaming)\.do[^\s"'<>()]*  # 다운로드 엔드포인트
    """ % (_EXT_RE, _EXT_RE)
)

# eGovFrame attachment file identifiers (commonly embedded in scripts / hidden inputs). When the audio
# is late-injected by a JS player (as on "The Scammer's Voice"), there is no static URL and only this
# file-id remains → reconstruct the URL via getFile.
EGOV_FILE_RE = re.compile(
    r"""(?xi)
    (?:atchFileId|atch_file_id|fileGrpId|file_grp_id)\s*[=:]\s*['"]?([A-Za-z0-9_\-]{6,})['"]?
    """
)
# Common endpoints that pull file lists / media via AJAX (hints for debug diagnostics).
AJAX_HINT_RE = re.compile(
    r"""(?xi)['"]([^\s"'<>]*(?:selectFileList|fileList|getFileList|selectMovie|media|player)[^\s"'<>]*\.(?:do|json)[^\s"'<>]*)['"]"""
)


def egov_getfile_urls(html: str) -> list[str]:
    """Best-effort reconstruction of getFile.do URLs from atchFileId/fileGrpId embedded in scripts/HTML."""
    urls = []
    for fid in dict.fromkeys(EGOV_FILE_RE.findall(html)):
        # fileSn ranges 0..N and is unknown, so only try 0 and 1 as candidates (for certainty, use the file-list API).
        for sn in ("0", "1"):
            urls.append(f"{BASE}/comm/getFile.do?atchFileId={fid}&fileSn={sn}")
    return urls


def scan_media_candidates(html: str, view_url: str = BASE) -> dict:
    """List every candidate that could be audio in a saved view HTML, grouped by category (offline diagnostics).

    Invoked via --analyze-html. Helps the DGX operator quickly pin down the actual audio
    injection pattern without live access."""
    soup = BeautifulSoup(html, "html.parser")
    out = {"media_tags": [], "data_attrs": [], "anchors": [], "ext_matches": [],
           "egov_file_ids": [], "ajax_endpoints": []}
    for tag in soup.find_all(["audio", "video", "source", "iframe", "embed", "object"]):
        for attr in ("src", "data-src", "data-audio", "data-file", "data-url", "data-mp3", "data"):
            if tag.get(attr):
                out["media_tags"].append(f"<{tag.name} {attr}={tag.get(attr)}>")
    for el in soup.find_all(True):
        for attr, v in el.attrs.items():
            if attr.startswith("data-") and isinstance(v, str) and any(x in v.lower() for x in (".mp3", ".mp4", ".wav", "getfile", "media", "stream")):
                out["data_attrs"].append(f"{el.name}[{attr}]={v}")
    for a in soup.find_all(["a", "button"]):
        blob = f'{a.get("href","")} {a.get("onclick","")}'.strip()
        if any(ext in blob.lower() for ext in AUDIO_EXT) or any(h in blob.lower() for h in DOWNLOAD_HINTS):
            out["anchors"].append(blob[:200])
    out["ext_matches"] = list(dict.fromkeys(m.group(0) for m in RAW_MEDIA_RE.finditer(html)))
    out["egov_file_ids"] = list(dict.fromkeys(EGOV_FILE_RE.findall(html)))
    out["ajax_endpoints"] = list(dict.fromkeys(m.group(1) for m in AJAX_HINT_RE.finditer(html)))
    return out


def parse_pages(spec: str) -> list[int]:
    """'1-20', '1,3,5', or '7' → list of page numbers."""
    pages: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            pages.extend(range(int(a), int(b) + 1))
        elif part:
            pages.append(int(part))
    return pages


def make_session(timeout: int) -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    s.request = _with_timeout(s.request, timeout)  # type: ignore
    return s


def _with_timeout(fn, timeout):
    def wrapped(method, url, **kw):
        kw.setdefault("timeout", timeout)
        return fn(method, url, **kw)
    return wrapped


def get(session: requests.Session, url: str, *, params=None, retries=4, **kw):
    """Retry with exponential backoff."""
    delay = 2.0
    last = None
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, **kw)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}"
            if r.status_code == 403:
                log.warning("403 Forbidden — check that this is running locally on the DGX (domestic network). url=%s", r.url)
        except requests.RequestException as e:
            last = str(e)
        if attempt < retries - 1:
            time.sleep(delay)
            delay *= 2
    log.error("GET failed (%s): %s", last, url)
    return None


def extract_ntt_ids(html: str) -> list[str]:
    """Broadly extract nttId candidates from list HTML (anchor href + onclick + JS)."""
    ids: list[str] = []
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a"):
        href = a.get("href", "") or ""
        onclick = a.get("onclick", "") or ""
        blob = f"{href} {onclick}"
        for m in re.finditer(r"nttId['\"]?\s*[:=,(]?\s*['\"]?(\d{2,})", blob):
            ids.append(m.group(1))
    # Also handle ids embedded in JS (patterns like fn_view(123))
    for m in re.finditer(r"fn_\w*[Vv]iew\w*\(\s*['\"]?(\d{2,})", html):
        ids.append(m.group(1))
    # Order-preserving dedup
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def extract_post(html: str, view_url: str) -> dict:
    """Extract title / body text / audio & attachment links from a detail-view HTML."""
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    for sel in ("h3", "h4", ".bbs_view .tit", ".view_title", "title"):
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            title = el.get_text(strip=True)
            break

    # Body text (text modality). Grab broadly, then normalize whitespace.
    body_el = (
        soup.select_one(".bbs_view")
        or soup.select_one(".view_cont")
        or soup.select_one("#content")
        or soup.body
    )
    text = re.sub(r"\n{3,}", "\n\n", body_el.get_text("\n", strip=True)) if body_el else ""

    audio: list[str] = []

    # 1) Media tags: <audio>/<video>/<source>/<iframe> + data-* attributes (handles JS players)
    for tag in soup.find_all(["audio", "video", "source", "iframe", "embed"]):
        for attr in ("src", "data-src", "data-audio", "data-file", "data-url", "data-mp3"):
            v = tag.get(attr)
            if v:
                audio.append(urljoin(view_url, v))

    # 2) Anchors/links: extension or download hint
    for a in soup.find_all(["a", "link", "button"]):
        href = a.get("href", "") or a.get("data-href", "") or ""
        if not href or href.startswith("javascript:void"):
            href = a.get("onclick", "") or ""
        low = href.lower()
        if any(ext in low for ext in AUDIO_EXT) or any(h in low for h in DOWNLOAD_HINTS):
            url = _resolve_download(href, view_url)
            if url:
                audio.append(url)

    # 3) Raw HTML regex scan — catches JS-injected / in-script strings that BeautifulSoup missed.
    for m in RAW_MEDIA_RE.finditer(html):
        audio.append(urljoin(view_url, m.group(0)))

    # 4) eGovFrame file-id reconstruction — for when the audio is late-injected by a JS player and
    #    has no static URL (the likely case for "The Scammer's Voice"). atchFileId/fileGrpId from
    #    scripts/hidden inputs → getFile.do.
    audio.extend(egov_getfile_urls(html))

    # Normalize + dedup
    audio = [u for u in dict.fromkeys(urljoin(view_url, a) for a in audio) if u]
    return {"title": title, "text": text, "audio": audio}


def _resolve_download(href: str, base_url: str) -> Optional[str]:
    """Best-effort reconstruction of the download URL from onclick/href."""
    # Patterns like fn_egov_downFile('FILE_ID','0')
    m = re.search(r"(?:atchFileId|fileId)['\"]?\s*[,=(]\s*['\"]([\w\-.]+)['\"]\s*,?\s*['\"]?(\d+)?", href)
    if m and "javascript" in href.lower():
        fid, sn = m.group(1), (m.group(2) or "0")
        return f"{BASE}/comm/getFile.do?atchFileId={fid}&fileSn={sn}"
    if href.lower().startswith(("http", "/")):
        return urljoin(base_url, href)
    return None


def _ext_from(ctype: str, url: str, name: str) -> str:
    """Guess the extension: filename → URL → Content-Type, in that order."""
    for cand in (name, urlparse(url).path):
        suf = Path(unquote(cand)).suffix.lower()
        if suf in AUDIO_EXT:
            return suf
    ct = ctype.split(";")[0].strip()
    return {
        "audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/wav": ".wav",
        "audio/x-wav": ".wav", "audio/mp4": ".m4a", "audio/aac": ".aac",
        "audio/ogg": ".ogg", "video/mp4": ".mp4", "video/webm": ".webm",
    }.get(ct, "")


def safe_filename(name: str, fallback: str, max_bytes: int = 180) -> str:
    """Sanitize a filename and truncate by UTF-8 byte length (extension preserved), to stay under the 255-byte limit."""
    name = (name or "").strip()
    if not name:
        return fallback
    stem = Path(name).stem
    ext = Path(name).suffix
    stem = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", stem).strip().strip(".")
    budget = max_bytes - len(ext.encode("utf-8"))
    enc = stem.encode("utf-8")[: max(8, budget)]
    stem = enc.decode("utf-8", "ignore").strip() or "audio"
    return f"{stem}{ext}" if ext else stem


def original_filename(r: requests.Response) -> str:
    """Extract the original filename from Content-Disposition and %-decode it (restores Korean characters)."""
    cd = r.headers.get("Content-Disposition", "")
    m = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)", cd)
    return unquote(m.group(1)).strip() if m else ""


def _stream_to_file(ar, post_dir: Path, fname: str) -> Path:
    """Save the stream to a file. If open() raises OSError (filename length etc.), the stream is unconsumed, so a fallback retry is possible."""
    dest = post_dir / fname
    with open(dest, "wb") as f:          # filename problems surface right here (stream not yet consumed)
        for chunk in ar.iter_content(8192):
            f.write(chunk)
    return dest


def fetch_list_page(session, board: Board, page: int) -> list[str]:
    r = get(session, board.list_url, params={"menuNo": board.menu_no, "pageIndex": page})
    if not r:
        return []
    ids = extract_ntt_ids(r.text)
    log.info("[%s] page %d → %d posts", board.label, page, len(ids))
    return ids


def process_post(session, board: Board, ntt_id: str, out_dir: Path, *, dry_run: bool,
                 save_html: bool = False) -> Optional[dict]:
    r = get(session, board.view_url, params={"nttId": ntt_id, "menuNo": board.menu_no})
    if not r:
        return None
    info = extract_post(r.text, r.url)
    rec = {"nttId": ntt_id, "board": board.board, "menuNo": board.menu_no,
           "source": board.label, "view_url": r.url, **info}

    # Diagnostics: if zero audio files were found (or --save-html), save the raw HTML under _debug/.
    # HTML is text, so it gets committed to git, letting us pin down the actual audio injection pattern remotely.
    if save_html or not info["audio"]:
        dbg = out_dir / "_debug"
        dbg.mkdir(parents=True, exist_ok=True)
        (dbg / f"{ntt_id}.html").write_text(r.text, encoding="utf-8")
        if not info["audio"]:
            log.warning("nttId=%s no audio found → saved raw HTML: %s", ntt_id, dbg / f"{ntt_id}.html")

    if dry_run:
        log.info("[dry] nttId=%s title=%r audio=%d", ntt_id, info["title"][:40], len(info["audio"]))
        return rec

    post_dir = out_dir / "posts" / ntt_id
    post_dir.mkdir(parents=True, exist_ok=True)
    (post_dir / "meta.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

    saved = []
    for i, url in enumerate(info["audio"]):
        ar = get(session, url, stream=True)
        if not ar:
            continue
        # Avoid accidentally saving an HTML page (store media/octet responses only).
        ctype = ar.headers.get("Content-Type", "").lower()
        if ctype.startswith("text/html"):
            log.warning("  skip (non-media response %s): %s", ctype, url)
            continue
        orig = original_filename(ar)                       # original (Korean) filename, decoded
        ext = _ext_from(ctype, url, orig) or ".bin"
        short = f"{ntt_id}_{i}{ext}"                        # deterministic short fallback name
        fname = safe_filename(orig, short)                 # sanitized + truncated
        try:
            dest = _stream_to_file(ar, post_dir, fname)
        except OSError as e:                               # filename length/character issue → short name
            log.warning("  filename issue (%s) → falling back to %s", e.__class__.__name__, short)
            try:
                dest = _stream_to_file(ar, post_dir, short)
                fname = short
            except OSError as e2:
                log.error("  save failed nttId=%s url=%s: %s", ntt_id, url, e2)
                continue
        saved.append({"file": fname, "original_name": orig, "url": url,
                      "bytes": dest.stat().st_size})
        log.info("  saved %s (%d bytes, %s)", fname, dest.stat().st_size, ctype or "?")
        time.sleep(0.3)
    rec["saved_files"] = saved
    return rec


_TURN_RE = re.compile(r"^\s*(사기범|피해자|상담원|고객|직원|안내|남자|여자)\s*[:：]")


def build_source_summary(out_dir: Path, board: Board, pages: str) -> dict:
    """Collected index.jsonl → **git-safe summary** (counts/titles/turn counts/extensions only; no raw text or audio).

    Raw audio and body text must not be committed (kept local), but the dev environment needs to see
    a summary of what was collected and how much.
    → auto-pushed as artifacts/fss_sources/<label>.json."""
    index_path = out_dir / "index.jsonl"
    posts = []
    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            try:
                posts.append(json.loads(line))
            except Exception:
                continue
    ext_counts: dict = {}
    total_bytes = 0
    n_with_media = 0
    post_rows = []
    for p in posts:
        saved = p.get("saved_files", []) or []
        text = p.get("text", "") or ""
        n_turns = sum(1 for ln in text.splitlines() if _TURN_RE.match(ln))
        exts = {}
        for s in saved:
            e = Path(s.get("file", "")).suffix.lower() or "?"
            exts[e] = exts.get(e, 0) + 1
            ext_counts[e] = ext_counts.get(e, 0) + 1
            total_bytes += int(s.get("bytes", 0) or 0)
        if saved:
            n_with_media += 1
        post_rows.append({"nttId": p.get("nttId"), "title": (p.get("title") or "")[:120],
                          "n_turns": n_turns, "n_media": len(saved), "exts": exts})
    return {
        "source": board.label, "board": board.board, "menuNo": board.menu_no,
        "media_kind": board.media, "pages": pages, "out_dir": str(out_dir),
        "n_posts": len(posts), "n_with_media": n_with_media,
        "n_media_files": sum(ext_counts.values()), "media_ext_counts": ext_counts,
        "total_media_bytes": total_bytes,
        "note": "git-safe 요약(원음성·본문 미포함). raw 는 DGX 로컬.",
        "posts": post_rows,
    }


def write_source_summary(out_dir: Path, board: Board, pages: str,
                         summary_dir: str = "artifacts/fss_sources") -> Path:
    summ = build_source_summary(out_dir, board, pages)
    sd = Path(summary_dir)
    sd.mkdir(parents=True, exist_ok=True)
    jp = sd / f"{board.label}.json"
    jp.write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("[%s] summary: posts=%d media_files=%d exts=%s bytes=%d → %s",
             board.label, summ["n_posts"], summ["n_media_files"], summ["media_ext_counts"],
             summ["total_media_bytes"], jp)
    return jp


def _selftest() -> int:
    """Validate the three parsers (list parsing / body & audio extraction / audio diagnostics) against synthetic HTML, no network needed."""
    list_html = """
    <ul class="bbs_list">
      <li><a href="view.do?nttId=36735&menuNo=200690">대출사기형(상세)</a></li>
      <li><a href="javascript:fn_view('36734')">수사기관 사칭형</a></li>
    </ul>"""
    view_html = """
    <html><head><title>대출사기형(상세) | 그놈 목소리</title></head>
    <body><div class="bbs_view"><h3>대출사기형(상세)</h3>
      <p>사기범 : 안녕하세요 고객님 김종현 대리입니다</p>
      <p>피해자 : 아네 대리님</p>
      <audio data-src="/cmm/fms/getImage.do?atchFileId=FILE_000123&fileSn=0"></audio>
      <script>var player = {atchFileId:'FILE_000999', fileSn:'0'};
              $.ajax({url:'/fss/bbs/selectFileList.do?nttId=36735'});</script>
    </div></body></html>"""

    ids = extract_ntt_ids(list_html)
    assert "36735" in ids and "36734" in ids, f"list parsing failed: {ids}"

    info = extract_post(view_html, f"{VIEW_URL}?nttId=36735")
    assert "김종현" in info["text"], "body text extraction failed"
    assert any("FILE_000123" in u for u in info["audio"]), f"data-src audio not detected: {info['audio']}"
    assert any("FILE_000999" in u for u in info["audio"]), f"script file-id not reconstructed: {info['audio']}"

    cand = scan_media_candidates(view_html)
    assert cand["egov_file_ids"], "egov file-id diagnostics failed"
    assert cand["ajax_endpoints"], "ajax endpoint diagnostics failed"

    # Segmentation (turn extraction) sanity — same rule as analyze_segmentation.
    turns = [ln for ln in info["text"].splitlines() if re.match(r"\s*(사기범|피해자)\s*[:：]", ln)]
    assert len(turns) == 2, f"turn parsing failed: {turns}"

    # Source presets / URL assembly sanity
    assert SOURCES["voice2"].board == "B0000207" and SOURCES["voice2"].menu_no == "200691"
    assert SOURCES["video"].board == "B0000203" and SOURCES["video"].media == "video"
    assert SOURCES["voice2"].list_url.endswith("/B0000207/list.do")
    assert SOURCES["voice"].default_out() == "data/raw/fss"
    assert SOURCES["voice2"].default_out() == "data/raw/fss/voice2"

    # Git-safe summary generation sanity (synthetic index.jsonl → summary; no raw audio/body text)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        od = Path(d)
        rec_a = {"nttId": "1", "title": "대출사기형", "text": "사기범 : 안녕\n피해자 : 네",
                 "saved_files": [{"file": "1_0.mp3", "bytes": 1000}]}
        rec_b = {"nttId": "2", "title": "사칭형", "text": "사기범 : 여보세요", "saved_files": []}
        (od / "index.jsonl").write_text(
            json.dumps(rec_a, ensure_ascii=False) + "\n" + json.dumps(rec_b, ensure_ascii=False) + "\n",
            encoding="utf-8")
        summ = build_source_summary(od, SOURCES["voice2"], "1-23")
        assert summ["n_posts"] == 2 and summ["n_with_media"] == 1
        assert summ["media_ext_counts"] == {".mp3": 1} and summ["total_media_bytes"] == 1000
        assert summ["posts"][0]["n_turns"] == 2 and "text" not in summ["posts"][0]  # body text not included
        assert summ["board"] == "B0000207"

    print("[selftest] OK — list parsing / body & audio extraction / diagnostics / turn parsing / source presets / git-safe summary passed")
    print(f"           detected audio candidates: {info['audio']}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Collector of real FSS voice-phishing recordings (run locally on the DGX).")
    ap.add_argument("--source", choices=sorted(SOURCES), default="voice",
                    help="Collection source: voice(B0000206) / voice2(B0000207) / video(B0000203 mp4)")
    ap.add_argument("--board", help="Specify a board id directly (e.g. B0000207) — arbitrary board instead of --source")
    ap.add_argument("--menu-no", help="Specify menuNo directly (together with --board)")
    ap.add_argument("--pages", default="1", help="Page range: '1-20' / '1,3,5' / '7'")
    ap.add_argument("--out", default=None, help="Output directory (default: automatic per source)")
    ap.add_argument("--summary", action="store_true", help="After collection, write a git-safe summary to artifacts/fss_sources/")
    ap.add_argument("--summary-only", action="store_true",
                    help="(Re)generate the summary from an existing index.jsonl without downloading")
    ap.add_argument("--dry-run", action="store_true", help="Only print what was discovered, without downloading")
    ap.add_argument("--resume", action="store_true", help="Skip nttIds already downloaded")
    ap.add_argument("--save-html", action="store_true",
                    help="Save every post's raw view HTML to _debug/ (for diagnosing audio patterns)")
    ap.add_argument("--sleep", type=float, default=1.0, help="Delay between requests (seconds, out of politeness)")
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--analyze-html", metavar="FILE",
                    help="Offline diagnosis of audio candidates in a saved view HTML (_debug/*.html); no network")
    ap.add_argument("--selftest", action="store_true", help="Offline parser self-test (no network)")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    if args.analyze_html:
        html = Path(args.analyze_html).read_text(encoding="utf-8", errors="ignore")
        cand = scan_media_candidates(html)
        print(f"=== audio candidate diagnostics: {args.analyze_html} ===")
        for k, vs in cand.items():
            print(f"\n[{k}] {len(vs)} found")
            for v in vs[:20]:
                print(f"   {v}")
        if not any(cand.values()):
            print("\n⚠️ Zero audio clues in the static HTML → the audio is likely late-injected via a separate AJAX call (fileList/media API).")
            print("   In the browser DevTools Network tab, check which .do/.json/.mp3 request URLs fire when opening the view page.")
        return 0

    # Source resolution: explicit --board (+ --menu-no) takes precedence, otherwise the --source preset.
    if args.board:
        if not args.menu_no:
            ap.error("--menu-no is required when using --board.")
        board = Board(args.board, args.menu_no, args.board.lower(), "audio")
    else:
        board = SOURCES[args.source]
    out_dir = Path(args.out or board.default_out())
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"
    log.info("source=%s board=%s menuNo=%s media=%s out=%s",
             board.label, board.board, board.menu_no, board.media, out_dir)

    # (Re)generate the summary only — from the existing index.jsonl, no network.
    if args.summary_only:
        write_source_summary(out_dir, board, args.pages)
        return 0

    done: set[str] = set()
    if args.resume and index_path.exists():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["nttId"])
            except Exception:
                pass
        log.info("resume: %d posts already indexed", len(done))

    session = make_session(args.timeout)

    all_ids: list[str] = []
    for page in parse_pages(args.pages):
        all_ids.extend(fetch_list_page(session, board, page))
        time.sleep(args.sleep)
    # dedup + resume filter
    seen = set()
    ids = [i for i in all_ids if not (i in seen or seen.add(i)) and i not in done]
    log.info("%d new posts queued for processing (dry_run=%s)", len(ids), args.dry_run)

    if not ids:
        log.warning("No nttIds to process. First check with --dry-run that list parsing works.")
        if not all_ids:
            log.warning("List parsing found 0 posts → the FSS HTML structure may have changed (extract_ntt_ids needs adjusting).")

    with open(index_path, "a", encoding="utf-8") as idx:
        for ntt_id in tqdm(ids, desc="posts"):
            rec = process_post(session, board, ntt_id, out_dir, dry_run=args.dry_run,
                               save_html=args.save_html)
            if rec and not args.dry_run:
                idx.write(json.dumps(rec, ensure_ascii=False) + "\n")
                idx.flush()
            time.sleep(args.sleep)

    log.info("Done. index=%s", index_path if not args.dry_run else "(dry-run, not written)")

    # After collection, write the git-safe summary (raw collection complete → auto-push target). Skipped for dry-run.
    if args.summary and not args.dry_run:
        write_source_summary(out_dir, board, args.pages)


if __name__ == "__main__":
    main()
