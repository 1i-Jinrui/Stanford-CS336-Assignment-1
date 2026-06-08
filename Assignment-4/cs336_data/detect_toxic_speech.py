import fasttext
from typing import Any
def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    model = fasttext.load_model('../local-shared-data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin')
    prediction = model.predict(text)

    label = prediction[0][0]   
    score = prediction[1][0]

    #去掉 __label__前缀
    toxic_speech_detect = label.replace('__label__', '')
    return (toxic_speech_detect, score)


