import json
import csv
import re
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="...")

parser.add_argument("-f", "--file", type=str, help="",default=0)

args = parser.parse_args()

input="detoxllm/"+args.file+"_generated_predictions.jsonl"
output="detoxllm/"+args.file+"_generated_predictions.csv"

print("input file: ",input)
print("output file: ",output)
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip(' \t\n\r\x0b\x0c\u200b')

def clean_prompt(raw_prompt):
    cleaned = re.sub(r'<\|.*?\|>', '', raw_prompt)
    start_index = cleaned.rfind(':') + 1
    cleaned = cleaned[start_index:] if start_index > 0 else cleaned
    cleaned = re.sub(r'^[.\s]+', '', cleaned)
    cleaned = re.sub(r'\s*assistant"?\s*$', '', cleaned, flags=re.IGNORECASE)
    return clean_text(cleaned)

def clean_label(raw_label):
    cleaned = raw_label.replace('<|eot_id|>', '')
    cleaned = re.sub(r'\s*-\s*', '-', cleaned)
    return clean_text(cleaned)

def jsonl_to_csv(input_path, output_path):
    if not Path(input_path).exists():
        raise FileNotFoundError(f"file {input_path} not exists")

    with open(input_path, 'r', encoding='utf-8') as jsonl_file, \
         open(output_path, 'w', newline='', encoding='utf-8') as csv_file:

        writer = csv.DictWriter(csv_file, fieldnames=['toxic', 'non-toxic', 'label'])
        writer.writeheader()

        for i, line in enumerate(jsonl_file, 1):
            try:
                data = json.loads(line.strip())
                processed = {
                    'toxic': clean_prompt(data['prompt']),
                    'non-toxic': clean_text(data['predict']),
                    'label': clean_label(data['label'])
                }
                writer.writerow(processed)
                
            except Exception as e:
                print(f"fail in row {i} : {str(e)}")
if __name__ == "__main__":
    try:
        jsonl_to_csv(input, output)
        print(f"success! file saved in {output}")
    except Exception as e:
        print(f"error: {str(e)}")