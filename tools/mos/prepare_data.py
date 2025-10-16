import os
import requests
from tqdm import tqdm

models = ["Zipvoice_vi", "F5TTS_vi"]
voices = ["Minh_Châu", "Minh_Ngọc", "Hảo", "Nhi", "Telesale_01"]
audio_dir = "data/audio"
os.makedirs(audio_dir, exist_ok=True)

with open("tools/mos/data/text.txt") as f:
    texts = [t.strip() for t in f.readlines() if t.strip() != ""]

with tqdm(total=len(texts) * len(models) * len(voices)) as pbar:
    for text in texts:
        for model in models:
            for voice in voices:
                data = dict(model=model, text=text, voice=voice)

                response = requests.post("http://localhost:5555/api/synthesize_prod", json=data)
                if response.status_code != 200:
                    print(response.json())
                    response.raise_for_status()

                filename = "__".join([model, voice, text.replace(" ", "_")[:20]])

                with open(os.path.join(audio_dir, filename) + ".wav", "wb") as f_out:
                    f_out.write(response.content)
                    
                pbar.update()
