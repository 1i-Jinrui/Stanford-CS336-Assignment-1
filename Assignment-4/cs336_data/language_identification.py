import fasttext
from typing import Any


def run_identify_language(text: str) -> tuple[Any, float]:

    model = fasttext.load_model('../local-shared-data/classifiers/lid.176.bin')
    # 2. 预测语言
    # fastText 返回格式: (['__label__en'], array([0.9876]))
    prediction = model.predict(text.replace('\n', ' '))# 因为一次只能预测一行文本，所以将换行符替换为空格
    print(prediction)
    
    label = prediction[0][0]
    score = prediction[1][0]

    #去掉 __label__前缀
    language_code = label.replace('__label__', '')
    return (language_code, score)
