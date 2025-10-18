import os
import random
import jsonlines
import subprocess
from tqdm import tqdm
from collections import defaultdict
import librosa
from quick_utils.common.mp_utils import execute_parallel


def worker(item: str, output_dir: str):
    audio_id, text, audio_path = item.strip().split("\t")
    filename = os.path.split(audio_path)[1]
    output_path = os.path.join(output_dir, filename)
    subprocess.check_call(["ffmpeg", "-loglevel", "panic", "-y", "-i", audio_path, "-ar", "24000", output_path])

    y, sr = librosa.load(audio_path, sr=None)
    assert sr == 22050
    src_dur = y.shape[0] / sr

    y, sr = librosa.load(output_path, sr=None)
    assert sr == 24000
    dst_dur = y.shape[0] / sr

    assert abs(dst_dur - src_dur) < 1e-4

    return "\t".join([audio_id, text, output_path])


def resample_sale_dataset():
    output_dir = "/data/longnh/data/telesale/audio"
    os.makedirs(output_dir, exist_ok=True)

    with open("data/vi/manifest/sale_metadata.tsv") as f_in, open("data/vi/manifest/sale_metadata_24khz.tsv", "w") as f_out:
        input_args = []
        for line in f_in.readlines():
            line = line.strip()
            input_args.append(dict(item=line, output_dir=output_dir))
        for line in execute_parallel(worker, input_args, max_workers=16):
            print(line, file=f_out)


def main():
    audio_prefix_dir = "/data/longnh/data"
    n_test = 10000
    n_train = 0
    n_skip = 0
    train_dur = 0
    test_dur = 0

    random.seed(8686)
    unique_file_name = set()

    with open("data/vi/manifest/train.tsv", "w") as f_train, open("data/vi/manifest/test.tsv", "w") as f_test:
        with jsonlines.open("/data/longnh/data/shared/filtered_by_wer=0.jsonl") as f_in:
            for item in tqdm(f_in):
                if item["duration"] < 1 or item["duration"] > 30:
                    n_skip += 1
                    continue

                audio_path = item["audio_filepath"]
                assert "audio_refined" in audio_path, audio_path
                assert item["offset"] is None, item["offset"]

                assert audio_path.startswith("../"), audio_path
                audio_path = os.path.join(audio_prefix_dir, audio_path[3:])
                assert os.path.exists(audio_path), audio_path

                filename = os.path.splitext(audio_path)[0]
                filename = os.path.split(filename)[1]
                
                if filename in unique_file_name:
                    n_skip += 1
                    continue
                
                unique_file_name.add(filename)

                if "Vivoice" in audio_path:
                    line = ["vivoice_" + filename, item["normed_text"], audio_path]
                elif "PhoAudioBook" in audio_path:
                    line = ["phoaudiobook_" + filename, item["normed_text"], audio_path]
                else:
                    raise NotImplementedError(audio_path)

                if n_test > 0 and random.random() < 0.1:
                    n_test -= 1
                    print("\t".join(line), file=f_test)
                    test_dur += item["duration"]
                else:
                    print("\t".join(line), file=f_train)
                    n_train += 1
                    train_dur += item["duration"]

            data_path = "data/vi/manifest/sale_metadata_24khz.tsv"
            speakers = ["minh_chau", "minh_ngoc"]
            split_by_speakers = defaultdict(list)
            duration_dict = {}

            with open(data_path) as f:
                for line in tqdm(f.readlines()):
                    found = False
                    _, _, audio_path = line.strip().split("\t")
                    assert os.path.exists(audio_path)
                    y, sr = librosa.load(audio_path, sr=None)
                    assert sr == 24000
                    duration_dict[audio_path] = y.shape[0] / sr
                    if duration_dict[audio_path] < 1 or duration_dict[audio_path] > 30:
                        n_skip += 1
                        continue

                    for spk in speakers:
                        if line.startswith(spk):
                            found = True
                            split_by_speakers[spk].append(line.strip())

            S = 10
            random.seed(8686)
            train = []
            test = []

            for k, vs in split_by_speakers.items():
                print(k, "=", len(vs))
                random.shuffle(vs)
                train.extend(vs[S:])
                test.extend(vs[:S])

            for line in train:
                print(line, file=f_train)
                _, _, audio_path = line.split("\t")
                n_train += 1
                train_dur += duration_dict[audio_path]

            for line in test:
                print(line, file=f_test)
                _, _, audio_path = line.split("\t")
                n_train += 1
                test_dur += duration_dict[audio_path]

    print("n_train =", n_train)
    print("n_skip =", n_skip)
    print("train_dur =", train_dur / 3600)
    print("test_dur =", test_dur / 3600)


if __name__ == "__main__":
    # resample_sale_dataset()
    main()
