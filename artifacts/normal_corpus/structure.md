# normal 코퍼스 구조 리포트

- root: `data/raw/normal`
- 총 파일(zip 내부 포함): **6438219**  |  zip 아카이브: 331
- 오디오: **8090개** (5GB) — ⚠️ 원본 비푸시(DGX 로컬)

## 파일 유형별 (개수 / 용량)
- `.json` : 6099522개 / 29GB
- `.txt` : 272388개 / 9GB
- `.tsv` : 58192개 / 207MB
- `.pcm` : 6000개 / 708MB
- `.wav` : 2090개 / 5GB
- `.xlsx` : 13개 / 7MB
- `.trn` : 12개 / 264MB
- `.md` : 1개 / 1KB
- `.irx934` : 1개 / 25MB

## 그래프 (SVG, zip 유지·git 안전)
- ![filetypes.svg](charts/filetypes.svg) `charts/filetypes.svg`
- ![by_corpus.svg](charts/by_corpus.svg) `charts/by_corpus.svg`
- ![by_corpus_bytes.svg](charts/by_corpus_bytes.svg) `charts/by_corpus_bytes.svg`
- ![register.svg](charts/register.svg) `charts/register.svg`
- ![speech_rate.svg](charts/speech_rate.svg) `charts/speech_rate.svg`

## 코퍼스별 상세 (최상위 폴더 — 하위폴더·zip 내부 귀속)
- **297.SNS 데이터 고도화** : 3225189파일 / 14GB · zip 2
  - .json×3225189
- **자유대화 음성(일반남녀)** : 2542359파일 / 2GB · zip 6
  - .json×2542359
- **020.주제별 텍스트 일상 대화 데이터** : 197304파일 / 1GB · zip 26
  - .json×98652, .txt×98652
- **141.한국어 멀티세션 대화** : 152000파일 / 3GB · zip 12
  - .txt×76000, .json×76000
- **011.일상대화 한국어 멀티세션 데이터** : 110710파일 / 15GB · zip 12
  - .txt×55355, .json×55355
- **021.용도별 목적대화 데이터** : 84762파일 / 386MB · zip 16
  - .json×42381, .txt×42381
- **046.공감형 대화** : 58062파일 / 309MB · zip 168
  - .tsv×29424, .json×28638
- **044.페르소나 대화** : 57536파일 / 402MB · zip 80
  - .tsv×28768, .json×28768
- **한국어 음성** : 6013파일 / 997MB · zip 1 · 오디오 6000
  - .pcm×6000, .trn×12, .irx934×1
- **Sample_voice** : 4000파일 / 427MB · zip 1 · 오디오 2000
  - .json×2000, .wav×2000
- **감정이 태깅된 자유대화_Sample** : 180파일 / 4GB · zip 0 · 오디오 90
  - .wav×90, .json×90
- **143.민원 업무 효율, 자동화를 위한 언어 AI 학습데이터** : 72파일 / 1GB · zip 4
  - .json×72
- **한국어 대화 요약** : 18파일 / 1GB · zip 2
  - .json×18
- **08.한국어대화** : 13파일 / 7MB · zip 1
  - .xlsx×13
- **README.md** : 1파일 / 1KB · zip 0
  - .md×1

## 오디오(wav) 길이 — RIFF 헤더 실측(benign 음성 발화 타이밍)
- WAV 파싱 2090개 → **총 ≈24.33시간**
- 샘플레이트: {16000: 2090} · 채널: {1: 2000, 2: 90}
- 파일당 초: 중앙 6.7 평균 41.9 p90 9.8 (min 2.0, max 1400.8, n=2090)
  · ✅ 헤더 실측(추정 아님). 오디오 샘플 미접근 → git 안전. 8초 세그먼트 = 15개면 2분.

## 오디오(pcm) 길이 추정 — benign 음성 발화 타이밍
- 가정 16000Hz/16bit/1ch → **총 ≈6.45시간**
- 파일당 초: 중앙 2.7 평균 3.9 p90 8.7 (min 0.3, max 20.1, n=6000)
  · ⚠️ pcm 은 헤더無 → 크기÷(rate·bytes·ch) 추정. 실 포맷은 데이터셋 스펙/tsv 로 검증(--pcm-rate 등 조정).
  · 세그먼트 8초 = 250KB/nibble.

## 폴더 트리 (루트 하위 상위 2단계, 파일수)
- `297.SNS 데이터 고도화/01-1.정식개방데이터` : 3225189
- `자유대화 음성(일반남녀)/Training` : 2278984
- `자유대화 음성(일반남녀)/Validation` : 263375
- `020.주제별 텍스트 일상 대화 데이터/01.데이터` : 197304
- `141.한국어 멀티세션 대화/01-1.정식개방데이터` : 152000
- `011.일상대화 한국어 멀티세션 데이터/3.개방데이터` : 110710
- `021.용도별 목적대화 데이터/01.데이터` : 84762
- `046.공감형 대화/01-1.정식개방데이터` : 58062
- `044.페르소나 대화/01-1.정식개방데이터` : 57536
- `한국어 음성/평가용_데이터` : 6000
- `Sample_voice/New_Sample.zip` : 4000
- `감정이 태깅된 자유대화_Sample/Sample` : 180
- `143.민원 업무 효율, 자동화를 위한 언어 AI 학습데이터/01.데이터` : 72
- `08.한국어대화/01_dialog` : 13
- `한국어 대화 요약/Training` : 9
- `한국어 대화 요약/Validation` : 9
- `한국어 음성/전시문_통합_스크립트` : 8
- `한국어 음성/_scripts` : 4
- `README.md` : 1
- `한국어 음성/한국어_음성_분야` : 1

## 오디오 포함 아카이브 (음성 위치)
- `Sample_voice/New_Sample.zip` → .json×2000, .wav×2000

## 타임스탬프 필드 탐지 (정상 클래스 발화속도·세그먼트 확보 가능성)
✅ 발견 → 실오디오 ASR 없이도 세그먼트 타이밍 확보 가능할 수 있음:
  - `sessionInfo.dialog.timestamp`

## 대화 register 통계 (피싱 FSS 와 비교 — 도메인 갭 판단)
- 스캔 json: 300개
- numberOfTurns_per_session(meta): 중앙 7 평균 7.1 p90 7 (min 7, max 9, n=599)
- utterances_per_session(parsed): 중앙 28 평균 28.4 p90 30 (min 28, max 34, n=300)
- words_per_utterance: 중앙 8 평균 9.4 p90 16 (min 2, max 84, n=8525)
- syllables_per_utterance: 중앙 23 평균 28.1 p90 48 (min 6, max 278, n=8525)
  · 참고 FSS 피싱: 단어/턴 중앙 4·평균 8.1, 2단어이하 38% (docs/01 §4.5).
  · ⚠️ 타임스탬프 없는 텍스트 대화면 발화속도는 오디오 ASR 필요(도메인 갭 주의).

## txt(평문 전사) register 통계 — 라인=발화 근사
- 스캔 txt: 300개
- lines_per_file: 중앙 1115 평균 1474.8 p90 2615 (min 732, max 10127, n=300)
- words_per_line: 중앙 2 평균 5.1 p90 14 (min 1, max 5887, n=442452)
- syllables_per_line: 중앙 0 평균 12.1 p90 41 (min 0, max 17057, n=442452)
  · 'speaker: ' 접두는 제거 후 발화만 계수. cf. FSS 피싱 단어/턴 중앙 4.

## tsv/csv 컬럼 스키마 (고유 구조)
- `르소나 대화/01-1.정식개방데이터/Training/01.원천데이터/TS_가족.zip!//Personas_가족_1000.tsv` (tab, 8컬럼, 20행)
  - 컬럼: id, utterance_id, text, bp_persona_info_id, bp_persona_profile_id, terminate, regDate, updDate
  - 발화컬럼: utterance_id
- tsv 발화 words_per_utterance: 중앙 1 평균 1.0 p90 1 (n=900)
- tsv 발화 syllables_per_utterance: 중앙 0 평균 0.0 p90 0 (n=900)

## xlsx 스키마 (zip 유지·메모리 파싱)
- `08.한국어대화/01_dialog/한국어대화_new_260226.zip!/B 의류(15,826)_new.xlsx`
  - 시트 ['Sheet1'] · 차원 A1:S15827 · 헤더: SPEAKER, SENTENCE, DOMAINID, DOMAIN, CATEGORY, SPEAKERID, SENTENCEID, MAIN, SUB, QA
- `08.한국어대화/01_dialog/한국어대화_new_260226.zip!/C 학원(4,773)_new.xlsx`
  - 시트 ['Sheet1'] · 차원 A1:T4774 · 헤더: SPEAKER, SENTENCE, DOMAINID, DOMAIN, CATEGORY, SPEAKERID, SENTENCEID, MAIN, SUB, QA
- `08.한국어대화/01_dialog/한국어대화_new_260226.zip!/D 소매점(14,949)_new.xlsx`
  - 시트 ['Sheet1'] · 차원 A1:S14950 · 헤더: SPEAKER, SENTENCE, DOMAINID, DOMAIN, CATEGORY, SPEAKERID, SENTENCEID, MAIN, SUB, QA
- `08.한국어대화/01_dialog/한국어대화_new_260226.zip!/E 생활서비스(11,087)_new.xlsx`
  - 시트 ['Sheet1'] · 차원 A1:S11088 · 헤더: SPEAKER, SENTENCE, DOMAINID, DOMAIN, CATEGORY, SPEAKERID, SENTENCEID, MAIN, SUB, QA
- `08.한국어대화/01_dialog/한국어대화_new_260226.zip!/F 카페(7,859)_new.xlsx`
  - 시트 ['Sheet1'] · 차원 A1:R7860 · 헤더: SPEAKER, SENTENCE, DOMAINID, DOMAIN, CATEGORY, SPEAKERID, SENTENCEID, MAIN, SUB, QA
- `08.한국어대화/01_dialog/한국어대화_new_260226.zip!/G 숙박업(7,113)_new.xlsx`
  - 시트 ['Sheet1'] · 차원 A1:S7114 · 헤더: SPEAKER, SENTENCE, DOMAINID, DOMAIN, CATEGORY, SPEAKERID, SENTENCEID, MAIN, SUB, QA
- `08.한국어대화/01_dialog/한국어대화_new_260226.zip!/H 관광여가오락(4,949)_new.xlsx`
  - 시트 ['Sheet1'] · 차원 A1:S4950 · 헤더: SPEAKER, SENTENCE, DOMAINID, DOMAIN, CATEGORY, SPEAKERID, SENTENCEID, MAIN, SUB, QA
- `08.한국어대화/01_dialog/한국어대화_new_260226.zip!/I 부동산(8,131)_new.xlsx`
  - 시트 ['Sheet1'] · 차원 A1:R8132 · 헤더: SPEAKER, SENTENCE, DOMAINID, DOMAIN, CATEGORY, SPEAKERID, SENTENCEID, MAIN, SUB, QA
- `08.한국어대화/01_dialog/한국어대화_new_260226.zip!/j 교통_최종본(250814).xlsx`
  - 시트 ['sheet1'] · 차원 A1:K668 · 헤더: 도메인, 화자, intent, subintent, question, 비식별 데이터, q_entity, a_entity, 용어사전, 지식베이스

## json 스키마 샘플

### `011.일상대화 한국어 멀티세션 데이터/3.개방데이터/1.데이터/Training/02.라벨링데이터/TL_session2.zip!//DAILY-004341-AP600017-WZ000005-02_03_02-S2.json`
```json
{
  "FileInfo": {
    "filename": "str",
    "sessionLevel": "str"
  },
  "participantsInfo": {
    "numberOfParticipants": "str",
    "speaker1": {
      "participantID": "str",
      "gender": "str",
      "age": "str",
      "occupation": "str",
      "bPlace": "str",
      "gPlace": "str",
      "rPlace": "str",
      "educationLevel": "str",
      "major": "str"
    },
    "speaker2": {
      "participantID": "str",
      "gender": "str",
      "age": "str",
      "occupation": "str",
      "bPlace": "str",
      "gPlace": "str",
      "rPlace": "str",
      "educationLevel": "str",
      "major": "str"
    }
  },
  "multisessionInfo": {
    "multisessionID": "str"
  },
  "personaInfo": {
    "apprenticeInfo": {
      "personaID": "str",
      "personaFeatures": [
        "<list len=3>",
        "str"
      ],
      "speakerType": "str"
    },
    "wizardInfo": {
      "personaID": "str",
      "personaFeatures": [
        "<list len=1>",
        "str"
      ],
      "speakerType": "str"
    }
  },
  "topicInfo": {
    "topicID": "str",
    "largeCategory": "str",
    "mediumCategory": "str",
    "smallCategory": "str"
  },
  "sessionInfo": [
    "<list len=2>",
    {
      "prevSessionID": "str",
      "prevTimeInfo": {
        "timeNum": "NoneType",
        "timeUnit": "NoneType"
      },
      "nthSession": "str",
      "numberOfUtterances": "str",
      "numberOfTurns": "str",
      "sessionID": "str",
      "sessionKeywords": [
        "<list len=1>",
        "str"

```

### `011.일상대화 한국어 멀티세션 데이터/3.개방데이터/1.데이터/Training/02.라벨링데이터/TL_session2.zip!//DAILY-004344-AP600019-WZ000006-02_03_02-S2.json`
```json
{
  "FileInfo": {
    "filename": "str",
    "sessionLevel": "str"
  },
  "participantsInfo": {
    "numberOfParticipants": "str",
    "speaker1": {
      "participantID": "str",
      "gender": "str",
      "age": "str",
      "occupation": "str",
      "bPlace": "str",
      "gPlace": "str",
      "rPlace": "str",
      "educationLevel": "str",
      "major": "str"
    },
    "speaker2": {
      "participantID": "str",
      "gender": "str",
      "age": "str",
      "occupation": "str",
      "bPlace": "str",
      "gPlace": "str",
      "rPlace": "str",
      "educationLevel": "str",
      "major": "str"
    }
  },
  "multisessionInfo": {
    "multisessionID": "str"
  },
  "personaInfo": {
    "apprenticeInfo": {
      "personaID": "str",
      "personaFeatures": [
        "<list len=3>",
        "str"
      ],
      "speakerType": "str"
    },
    "wizardInfo": {
      "personaID": "str",
      "personaFeatures": [
        "<list len=1>",
        "str"
      ],
      "speakerType": "str"
    }
  },
  "topicInfo": {
    "topicID": "str",
    "largeCategory": "str",
    "mediumCategory": "str",
    "smallCategory": "str"
  },
  "sessionInfo": [
    "<list len=2>",
    {
      "prevSessionID": "str",
      "prevTimeInfo": {
        "timeNum": "NoneType",
        "timeUnit": "NoneType"
      },
      "nthSession": "str",
      "numberOfUtterances": "str",
      "numberOfTurns": "str",
      "sessionID": "str",
      "sessionKeywords": [
        "<list len=1>",
        "str"

```

### `011.일상대화 한국어 멀티세션 데이터/3.개방데이터/1.데이터/Training/02.라벨링데이터/TL_session2.zip!//DAILY-003830-AP600008-WZ000006-02_03_02-S2.json`
```json
{
  "FileInfo": {
    "filename": "str",
    "sessionLevel": "str"
  },
  "participantsInfo": {
    "numberOfParticipants": "str",
    "speaker1": {
      "participantID": "str",
      "gender": "str",
      "age": "str",
      "occupation": "str",
      "bPlace": "str",
      "gPlace": "str",
      "rPlace": "str",
      "educationLevel": "str",
      "major": "str"
    },
    "speaker2": {
      "participantID": "str",
      "gender": "str",
      "age": "str",
      "occupation": "str",
      "bPlace": "str",
      "gPlace": "str",
      "rPlace": "str",
      "educationLevel": "str",
      "major": "str"
    }
  },
  "multisessionInfo": {
    "multisessionID": "str"
  },
  "personaInfo": {
    "apprenticeInfo": {
      "personaID": "str",
      "personaFeatures": [
        "<list len=3>",
        "str"
      ],
      "speakerType": "str"
    },
    "wizardInfo": {
      "personaID": "str",
      "personaFeatures": [
        "<list len=1>",
        "str"
      ],
      "speakerType": "str"
    }
  },
  "topicInfo": {
    "topicID": "str",
    "largeCategory": "str",
    "mediumCategory": "str",
    "smallCategory": "str"
  },
  "sessionInfo": [
    "<list len=2>",
    {
      "prevSessionID": "str",
      "prevTimeInfo": {
        "timeNum": "NoneType",
        "timeUnit": "NoneType"
      },
      "nthSession": "str",
      "numberOfUtterances": "str",
      "numberOfTurns": "str",
      "sessionID": "str",
      "sessionKeywords": [
        "<list len=1>",
        "str"

```

> 샘플 파일: `artifacts/normal_corpus/samples/` (텍스트 최대 500자, 3개/유형). 라이선스상 전체 전사 비공개 — 형식 확인용 소량만.