import fitz
import requests
import json
pdf_file_path = '/home/makoto/Document/Anaconda/GNN-DGLPro/inzemi/2206.00272.pdf'
output_file_path = "output16.md"
page_num = 1
print('開始')
with fitz.open(pdf_file_path) as pdf_in:

    text = ""
    for page in pdf_in:
        page1 = page.get_text()

        data = {
        'auth_key': 'e2a8d334-1e32-b537-03a6-35731578b13e:fx',
        'text': page1.replace('-\n','').replace('\n',''),
        'target_lang': 'JA'
        }

        response = requests.post('https://api-free.deepl.com/v2/translate', data=data)
        d = json.loads(response.text)
        _text = f"""### Page{page_num}\n{d['translations'][0]['text']}\n"""
        text = text + _text
        page_num += 1
print('書き込み')
with open(output_file_path, 'w',encoding='utf-8') as file:
        file.write(text)