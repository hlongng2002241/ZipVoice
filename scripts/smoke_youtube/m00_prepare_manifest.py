"""
Sample a handful of short utterances from the user's real YouTube ASR-transcript
corpus (schema: {"id", "audio_path", "duration", "sample_rate", "text",
"language"}) and build a lhotse CutSet directly (not via the TSV-only
zipvoice.bin.prepare_dataset, which has no language column), preserving each
utterance's "language" field on its SupervisionSegment -- needed to verify
train_zipvoice.py's per-item [LANG:xx]/dropout derivation (see
tokenize_text()) end-to-end, since that reads supervision.language directly.
"""

import json
import os

from lhotse import CutSet, Recording, SupervisionSegment

SRC = "/data4/audio/youtube/exp/01/audio.exist.denoise.info.vol.diarize.mos.asr.abs.final.jsonl"
OUT_DIR = "data/smoke_youtube/manifests"
N_TRAIN = 35
# dev_dataloaders' DynamicBucketingSampler uses lhotse's default num_buckets
# (10), which asserts num_buckets <= num_cuts -- so the dev set needs >= 10.
N_DEV = 10
MAX_DUR = 12.0
MIN_DUR = 1.0


def build_cuts(rows):
    cuts = []
    for row in rows:
        recording = Recording.from_file(row["audio_path"], recording_id=row["id"])
        supervision = SupervisionSegment(
            id=row["id"],
            recording_id=row["id"],
            start=0.0,
            duration=recording.duration,
            text=row["text"],
            language=row.get("language"),
        )
        cut = recording.to_cut()
        cut.supervisions = [supervision]
        cuts.append(cut)
    return CutSet.from_cuts(cuts)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    picked = []
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            dur = obj.get("duration")
            path = obj.get("audio_path")
            if dur is None or not (MIN_DUR <= dur <= MAX_DUR):
                continue
            if not path or not os.path.isfile(path):
                continue
            picked.append(obj)
            if len(picked) >= N_TRAIN + N_DEV:
                break

    assert len(picked) >= N_TRAIN + N_DEV, (
        f"only found {len(picked)} usable utterances, need {N_TRAIN + N_DEV}"
    )

    train_rows = picked[:N_TRAIN]
    dev_rows = picked[N_TRAIN : N_TRAIN + N_DEV]

    for name, rows in [("train", train_rows), ("dev", dev_rows)]:
        cuts = build_cuts(rows)
        out_path = os.path.join(OUT_DIR, f"smoke_youtube_cuts_{name}.jsonl.gz")
        cuts.to_file(out_path)
        langs = {c.supervisions[0].language for c in cuts}
        print(f"Wrote {len(cuts)} cuts to {out_path} (languages: {langs})")


if __name__ == "__main__":
    main()
