import requests


for model in ["Zipvoice_vi", "F5TTS_vi"]:
    data = dict(
        model=model,
        text="hôm nay tôi buồn một mình trên phố đông, nơi ánh nhìn ánh mắt long lanh.",
        voice="Minh_Châu",
    )

    response = requests.post("http://localhost:5555/api/synthesize_prod", json=data)
    if response.status_code != 200:
        print(response.json())
        response.raise_for_status()

    with open(f"temp/audio_{data['model']}.wav", "wb") as f:
        f.write(response.content)
