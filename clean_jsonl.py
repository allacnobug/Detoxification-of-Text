import json
import csv
import re
from pathlib import Path
import argparse

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="...")

# 添加参数
parser.add_argument("-f", "--file", type=str, help="",default=0)

# 解析参数
args = parser.parse_args()

input="detoxllm/"+args.file+"_generated_predictions.jsonl"
output="detoxllm/"+args.file+"_generated_predictions.csv"
# 使用参数
print("输入文件",input)
print("输出文件",output)
def clean_text(text):
    """通用文本清洗：去除多余空格和特殊标记"""
    # 合并多个空格为单个空格
    text = re.sub(r'\s+', ' ', text)
    # 去除首尾空格和特殊不可见字符
    return text.strip(' \t\n\r\x0b\x0c\u200b')

def clean_prompt(raw_prompt):
    """专用prompt清洗逻辑"""
    # 移除固定前缀和XML标签
    cleaned = re.sub(r'<\|.*?\|>', '', raw_prompt)
    # 定位实际句子开始位置
    start_index = cleaned.rfind(':') + 1
    # 提取有效内容
    cleaned = cleaned[start_index:] if start_index > 0 else cleaned
    # 去除开头多余的标点和空格
    cleaned = re.sub(r'^[.\s]+', '', cleaned)
    cleaned = re.sub(r'\s*assistant"?\s*$', '', cleaned, flags=re.IGNORECASE)
    return clean_text(cleaned)

def clean_label(raw_label):
    """专用label清洗逻辑"""
    # 移除结尾的特殊标记
    cleaned = raw_label.replace('<|eot_id|>', '')
    # 处理连接符周围的空格
    cleaned = re.sub(r'\s*-\s*', '-', cleaned)
    return clean_text(cleaned)

def jsonl_to_csv(input_path, output_path):
    if not Path(input_path).exists():
        raise FileNotFoundError(f"输入文件 {input_path} 不存在")

    with open(input_path, 'r', encoding='utf-8') as jsonl_file, \
         open(output_path, 'w', newline='', encoding='utf-8') as csv_file:

        writer = csv.DictWriter(csv_file, fieldnames=['toxic', 'non-toxic', 'label'])
        writer.writeheader()

        for i, line in enumerate(jsonl_file, 1):
            try:
                data = json.loads(line.strip())
                # 清洗各字段
                processed = {
                    'toxic': clean_prompt(data['prompt']),
                    'non-toxic': clean_text(data['predict']),
                    'label': clean_label(data['label'])
                }
                writer.writerow(processed)
                
            except Exception as e:
                print(f"第 {i} 行处理失败: {str(e)}")
                print(f"问题数据: {line[:100]}...")  # 打印前100字符避免输出过长
if __name__ == "__main__":
    try:
        jsonl_to_csv(input, output)
        print(f"转换完成！清洗后的文件保存在: {output}")
    except Exception as e:
        print(f"转换失败: {str(e)}")