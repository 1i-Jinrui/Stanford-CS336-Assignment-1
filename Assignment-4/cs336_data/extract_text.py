from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding

def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    
    try: # 尝试使用 UTF-8 解码
        html_str = html_bytes.decode("utf-8")
    except UnicodeDecodeError: # 如果 UTF-8 解码失败，尝试使用检测到的编码
        encoding = detect_encoding(html_bytes)
        html_str = html_bytes.decode(encoding)
    
    text = extract_plain_text(html_str)
    return text