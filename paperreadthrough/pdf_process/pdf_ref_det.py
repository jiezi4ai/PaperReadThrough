import re
import math
import string
import json
from typing import List, Dict, Optional
from thefuzz import fuzz # pip install thefuzz  https://github.com/seatgeek/thefuzz
from json_repair import repair_json  # https://github.com/mangiucugna/json_repair/

from apis.semanticscholar_tool import SemanticScholarKit
from paper_comprehension.prompts import reference_example_json, extract_ref_prompt
from paper_comprehension.models import llm_gen_w_retry

def remove_non_text_chars(text, with_digits: Optional[bool]=True):
    """remove non text chars
    """
    valid_chars = string.ascii_letters
    if with_digits == True:
        valid_chars += string.digits  # 包含所有字母和数字的字符串
    cleaned_text = ''
    for char in text:
        if char in valid_chars:
            cleaned_text += char
    return cleaned_text


def text_match(text_a, text_b, with_digits: Optional[bool]=True):
    """"fuzzy match between text_a and text_b"""
    text_a = remove_non_text_chars(text_a, with_digits).lower()
    text_b = remove_non_text_chars(text_b, with_digits).lower()
    return fuzz.ratio(text_a, text_b)


class PDFRef:
    def __init__(self, pdf_json):
        self.pdf_json = pdf_json

    def find_ref_text(self, reference_metadata):
        """"find ref text position"""
        start_pos = 0
        end_pos = len(self.pdf_json)

        # search for reference title postion as start 
        for i in range(len(self.pdf_json)):
            if self.pdf_json[i].get('text_level') == 1:
                if text_match(self.pdf_json[i].get('text'), 'Reference', False) > 90:
                    start_pos = i + 1
                    break

        if start_pos > 0:
            for j in range(start_pos, len(self.pdf_json)):
                if self.pdf_json[j].get('text_level') is not None:
                    end_pos = j
                    break

        for idx in range(start_pos, end_pos):
            item = self.pdf_json[idx]
            item['type'] = "reference"


    def identify_ref_metadata(self, api_key, model_name):
        lines = []
        for idx, item in enumerate(self.pdf_json):
            if item['type'] == "reference":
                line = {'line_id': idx, 'bib_text': item.get('text')}
                lines.append(line)
        
        # use LLM to identify ref info
        n = math.ceil(len(lines) / 20)
        results = []
        for i in range(n):
            input_text = lines[i*20:i*20+20]
            qa_prompt = extract_ref_prompt.format(
                reference_example_json=reference_example_json,
                input_text=input_text)
            res = llm_gen_w_retry(api_key, model_name, qa_prompt)
            results.append(res)

        # extract llm results
        doi_arxiv_ids = []
        for res in results:
            res_json = (json.loads(repair_json(res)))
            for item in res_json:
                if item.get('doi') is not None:
                    doi_arxiv_ids.append(item.get('doi'))
                elif item.get('url', '').startswith("https://arxiv.org"):
                    arxiv_no = item.get('url', '').split('/')[-1]
                    arxiv_id = re.sub(r'v\d+$', '', arxiv_no)
                    doi = f"10.48550/arXiv.{arxiv_id}"
                    doi_arxiv_ids.append(doi)

        # get metadata from s2
        s2 = SemanticScholarKit()
        reference_meatadata = s2.search_paper_by_ids(id_list=doi_arxiv_ids)
        return reference_meatadata


    def align_ref_data():
        """"identify reference items in pdf content json"""
        for idx in range(start_pos, end_pos):
            item = self.pdf_json[idx]
            if item.get('type') == 'text' and len(item.get('text')) < 500:
                if len(reference_metadata) > 0:
                    for ref in reference_metadata:
                        title = ref.get('citedPaper', {}).get('title')
                        ss_paper_id = ref.get('citedPaper', {}).get('paperId')
                        if title:
                            match_ratio = text_patial_match(title, item.get('text'), True) 
                            if match_ratio > 80:
                                item['if_aligned'] = True
                                item['type'] = "reference"
                                item['ss_paper_id'] = ss_paper_id
                                break
                else:
                    if start_pos > 0 and end_pos < len(self.pdf_json) and start_pos < end_pos:
                        item['if_aligned'] = True
                        item['type'] = "reference"
                        item['ss_paper_id'] = None