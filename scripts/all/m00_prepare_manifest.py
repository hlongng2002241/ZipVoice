"""
Combine the full YouTube + TopTop corpora into one training pool and split
off held-out test sets: 1000 Vietnamese + 500 English utterances (counts
requested by the user), everything else goes to train.

Sources:
- YouTube: schema {"id", "audio_path", "duration", "sample_rate", "text",
  "language"} -- language already present (only "Vietnamese"/"English" occur
  in this corpus, confirmed by scanning all 52,734 records).
- TopTop: schema {"id", "audio_path", "duration", "sample_rate", "text",
  "parent_id"} -- no "language" field. Per the user: every TopTop sample is
  Vietnamese, so "Vietnamese" is assigned to all 120,727 records here (not by
  editing the source file, which is outside this project's control).

Builds lhotse Recording/SupervisionSegment/Cut objects directly from the
JSONL's own duration/sample_rate fields (no per-file audio-header probing),
since scanning ~173K files that way would be slow and the JSONL's reported
values are already the source pipeline's own output.

Output ("metadata", per the user -- these are lhotse manifests, not audio):
  data/all/manifests/train.jsonl.gz     -- everything not held out
  data/all/manifests/test_vi.jsonl.gz   -- 1000 held-out Vietnamese utterances
  data/all/manifests/test_en.jsonl.gz   -- 500 held-out English utterances
  data/all/manifests/test.jsonl.gz      -- test_vi + test_en combined, used
                                            as --dev-manifest for training
  data/all/split_summary.json           -- exact counts, for reproducibility
"""

import json
import os
import random
import sys

# Run as a plain script from anywhere -- put the repo root (two levels up
# from scripts/all/) on sys.path so `import zipvoice...` resolves without
# needing `pip install -e .` or `python -m`.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lhotse import CutSet, Recording, SupervisionSegment
from lhotse.audio import AudioSource

from zipvoice.tokenizer.multilingual_tokenizer import normalize_language_name

YOUTUBE_PATH = "/data4/audio/youtube/exp/01/audio.exist.denoise.info.vol.diarize.mos.asr.abs.final.jsonl"
TOPTOP_PATH = "/data4/audio/toptop/exp/04/metadata.dur.norm.cut.dur.sv.wer.punc.hard_merge.dur.wer.dur.flac.jsonl"

OUT_DIR = "data/all/manifests"
SUMMARY_PATH = "data/all/split_summary.json"

N_TEST_VI = 1000
N_TEST_EN = 500
SEED = 42


def iter_records(path, default_language=None):
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("text")
            duration = obj.get("duration")
            audio_path = obj.get("audio_path")
            if not text or not duration or duration <= 0 or not audio_path:
                continue
            if not os.path.isfile(audio_path):
                continue
            language = obj.get("language") or default_language
            yield {
                "id": obj["id"],
                "audio_path": audio_path,
                "duration": float(duration),
                "sample_rate": obj.get("sample_rate", 24000),
                "text": text,
                "language": language,
            }


def make_cut(row):
    sampling_rate = row["sample_rate"]
    num_samples = int(round(row["duration"] * sampling_rate))
    recording = Recording(
        id=row["id"],
        sources=[AudioSource(type="file", channels=[0], source=row["audio_path"])],
        sampling_rate=sampling_rate,
        num_samples=num_samples,
        duration=row["duration"],
    )
    supervision = SupervisionSegment(
        id=row["id"],
        recording_id=row["id"],
        start=0.0,
        duration=row["duration"],
        text=row["text"],
        language=row["language"],
    )
    cut = recording.to_cut()
    cut.supervisions = [supervision]
    return cut


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(SEED)

    print("Reading YouTube...")
    youtube_rows = list(iter_records(YOUTUBE_PATH))
    print(f"  usable YouTube rows: {len(youtube_rows)}")

    print("Reading TopTop (assigning language='Vietnamese' to all)...")
    toptop_rows = list(iter_records(TOPTOP_PATH, default_language="Vietnamese"))
    print(f"  usable TopTop rows: {len(toptop_rows)}")

    all_rows = youtube_rows + toptop_rows

    by_lang = {"vi": [], "en": [], "other": []}
    invalid_rows = []
    for row in all_rows:
        norm = normalize_language_name(row["language"])
        if norm is None:
            invalid_rows.append(row)
            continue
        by_lang.setdefault(norm if norm in ("vi", "en") else "other", []).append(row)

    if invalid_rows:
        examples = ", ".join(
            f"{r['id']!r} (language={r['language']!r})" for r in invalid_rows[:10]
        )
        raise ValueError(
            f"{len(invalid_rows)} row(s) have a missing or unrecognized "
            f"language and cannot be used for training -- every utterance "
            f"must have a valid language before training. Examples: {examples}"
        )

    for bucket in by_lang.values():
        rng.shuffle(bucket)

    assert len(by_lang["vi"]) >= N_TEST_VI, (
        f"only {len(by_lang['vi'])} Vietnamese rows available, need {N_TEST_VI}"
    )
    assert len(by_lang["en"]) >= N_TEST_EN, (
        f"only {len(by_lang['en'])} English rows available, need {N_TEST_EN}"
    )

    test_vi_rows = by_lang["vi"][:N_TEST_VI]
    test_en_rows = by_lang["en"][:N_TEST_EN]
    train_rows = by_lang["vi"][N_TEST_VI:] + by_lang["en"][N_TEST_EN:] + by_lang["other"]
    rng.shuffle(train_rows)

    splits = {
        "train": train_rows,
        "test_vi": test_vi_rows,
        "test_en": test_en_rows,
    }
    for name, rows in splits.items():
        cuts = CutSet.from_cuts(make_cut(row) for row in rows)
        out_path = os.path.join(OUT_DIR, f"{name}.jsonl.gz")
        cuts.to_file(out_path)
        print(f"Wrote {len(rows)} cuts to {out_path}")

    test_all = CutSet.from_cuts(
        make_cut(row) for row in (test_vi_rows + test_en_rows)
    )
    test_all_path = os.path.join(OUT_DIR, "test.jsonl.gz")
    test_all.to_file(test_all_path)
    print(f"Wrote {len(test_vi_rows) + len(test_en_rows)} cuts to {test_all_path}")

    summary = {
        "seed": SEED,
        "youtube_usable_rows": len(youtube_rows),
        "toptop_usable_rows": len(toptop_rows),
        "counts_by_language_before_split": {k: len(v) for k, v in by_lang.items()},
        "train": len(train_rows),
        "test_vi": len(test_vi_rows),
        "test_en": len(test_en_rows),
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote split summary to {SUMMARY_PATH}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
