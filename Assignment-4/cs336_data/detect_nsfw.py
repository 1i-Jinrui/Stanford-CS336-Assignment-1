import fasttext
from typing import Any

def run_classify_nsfw(text: str) -> tuple[Any, float]:
    model = fasttext.load_model('../local-shared-data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin')
    prediction = model.predict(text)

    label = prediction[0][0]   
    score = prediction[1][0]

    #去掉 __label__前缀
    nsfw_detect = label.replace('__label__', '')
    return (nsfw_detect, score)


