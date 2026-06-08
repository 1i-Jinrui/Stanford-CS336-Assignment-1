import re

def run_mask_ips(text: str) -> tuple[str, int]:
    # ip就是4个数字，中间用"."隔开，e.g. 192.168.0.1 
    # 本来可以使用ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'，但是它也会匹配非法 IP
    # 它只限制了“每段是 1 到 3 位数字”，没有限制每段必须在 0-255 之间。
    # 匹配 0-255 的正则表达式
    octet = r'(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)'
    
    # 完整的IP地址模式
    ip_pattern = rf'\b{octet}\.{octet}\.{octet}\.{octet}\b'

    masked_text, count = re.subn(
        pattern=ip_pattern,
        repl='|||IP_ADDRESS|||',
        string=text
    )
    
    return (masked_text, count)

