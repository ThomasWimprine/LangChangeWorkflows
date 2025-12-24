import json
import os
import jsonl

data = []

if os.path.exists('result.json'):
    with open('result.json', 'r') as f:
        jsonl.dump(data, f)
    # print(data)
else:
    print("File 'result.json' does not exist.")
    print("No data to display.")

print(data)
