# paper pdf post process
# ideally the paper table of content (toc), paper markdown text, paper content list of json are ready
# refer to mineru_tool.py for pdf processing 
import re 
from typing import List, Dict, Optional

import copy
import string
from bs4 import BeautifulSoup
from thefuzz import fuzz # pip install thefuzz  https://github.com/seatgeek/thefuzz

from pdf_process import APPENDDIX_TITLES, IMG_REGX_NAME_PTRN, TBL_REGX_NAME_PTRN, EQT_REGX_NAME_PTRN


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


def text_patial_match(shorter_text, longer_text, with_digits: Optional[bool]=True):
    """"partial fuzzy match between text_a and text_b"""
    shorter_text = remove_non_text_chars(shorter_text, with_digits).lower()
    longer_text = remove_non_text_chars(longer_text, with_digits).lower()
    return fuzz.partial_ratio(shorter_text, longer_text)


class PDFProcess:
    def __init__(self, pdf_path, pdf_toc, pdf_json):
        """load pdf related files and data
        Args:
            pdf_path: path to pdf file
            pdf_toc: table of content genreated from PDFOutline class
            pdf_json: json content from MinerU after processing PDF ("_content_list.json")
        """
        self.pdf_path = pdf_path
        self.pdf_toc = pdf_toc
        self.pdf_json = pdf_json

    # match title information from content list to that from PDF ToC
    def align_content_toc(self):
        """match title information from content list to that from PDF ToC"""
        mtched_toc_idx = []
        for idx1, item1 in enumerate(self.pdf_json):  # enumerate content json for titles
            if item1.get('type') == 'text' and item1.get('text_level') is not None:
                item1_page_idx = item1.get('page_idx')
                item1_title = item1.get('text')
                item1_title = re.sub(r"^[A-Za-z]\.", "", item1_title)
                
                pattern = '|'.join(re.escape(title) for title in APPENDDIX_TITLES) 
                if re.search(pattern, item1_title, re.IGNORECASE):
                    item1['type'] = 'title'
                    item1['if_aligned'] = True
                    item1['text_level'] = 1
                    item1['aligned_text'] = item1_title
                    item1['if_appendix'] = True
                    item1['if_collapse'] = False
                    continue

                for idx2, item2 in enumerate(self.pdf_toc):  # enumerate pdf toc 
                    if idx2 not in mtched_toc_idx:
                        item2_title = item2.get('title')
                        item2_title = re.sub(r"^[A-Za-z]\.", "", item2_title)
                        item2_page_idx = item2.get('page')

                        if item1_page_idx == item2_page_idx or item1_page_idx + 1 == item2_page_idx:  # titles of the same page
                            match_ratio = text_match(item1_title, item2_title, False)
                            if match_ratio > 90:  # confirmed title
                                item1['type'] = 'title'
                                item1['if_aligned'] = True
                                item1['text_level'] = item2.get('level')
                                item1['aligned_text'] = f"{item2['nameddest']} {item2_title}"
                                item1['if_appendix'] = item2.get('if_appendix')
                                item1['if_collapse'] = item2.get('if_collapse')
                                mtched_toc_idx.append(idx2)
                                break

    def align_content_images(self):
        """align and standardize image information"""
        img_id_lst = []  # store all images ids

        for idx, item in enumerate(self.pdf_json):
            if item['type'] in ['image']:
                if item['img_caption'] == [] and item['img_footnote'] == []:  # without caption and footnote in the block
                    prev_pos, next_pos = 9999, 9999
                    prev_id, next_id = None, None
                    prev_img_ids, next_img_ids = [], []

                    # check next block for image description info
                    if idx < len(self.pdf_json) - 1 and self.pdf_json[idx+1]['type'] == 'text' and self.pdf_json[idx+1].get('text') is not None:
                        next_item = self.pdf_json[idx+1]
                        mtch_rslts = re.finditer(IMG_REGX_NAME_PTRN, next_item['text'], re.IGNORECASE)

                        next_img_ids = []
                        for match in mtch_rslts:
                            next_img_ids.append(match.group(0)) 

                        for id in next_img_ids:
                            if id not in img_id_lst:
                                next_id = id
                                next_pos = next_item['text'].index(id)
                                next_img_ids = next_img_ids.remove(id)
                                break

                    # check previous block for image description info
                    if idx > 1 and self.pdf_json[idx-1]['type'] == 'text' and self.pdf_json[idx-1].get('text') is not None:
                        prev_item = self.pdf_json[idx-1]
                        mtch_rslts = re.finditer(IMG_REGX_NAME_PTRN, prev_item['text'], re.IGNORECASE)

                        prev_img_ids = []
                        for match in mtch_rslts:
                            prev_img_ids.append(match.group(0)) 

                        for id in prev_img_ids:
                            if id not in img_id_lst:
                                prev_id = id
                                prev_pos = prev_item['text'].index(id)
                                prev_img_ids = prev_img_ids.remove(id)
                                break
                    
                    if next_pos < prev_pos:
                        item['id'] = next_id
                        item['related_ids'] = next_img_ids
                        item['if_aligned'] = True
                        item['img_caption'] = [next_item['text']]
                        next_item['if_aligned'] = False

                    elif prev_pos < next_pos:
                        item['id'] = prev_id
                        item['related_ids'] = prev_img_ids
                        item['if_aligned'] = True
                        item['img_caption'] = [prev_item['text']]
                        prev_item['if_aligned'] = False
                        
                else:
                    desc = "\n".join(item.get('img_caption', [])) + "\n" + "\n".join(item.get('img_footnote', []))
                    mtch_rslts = re.finditer(IMG_REGX_NAME_PTRN, desc, re.IGNORECASE)

                    img_ids = []
                    for match in mtch_rslts:
                        img_ids.append(match.group(0)) 
                    
                    for id in img_ids:
                        if id not in img_id_lst:
                            break

                    item['id'] = id
                    item['related_ids'] = img_ids.remove(id)
                    item['if_aligned'] = True


    def align_content_images(self):
        """align and standardize image information"""
        tbl_id_lst = []  # store all table ids

        for idx, item in enumerate(self.pdf_json):
            if item['type'] in ['table']:
                if item['table_caption'] == [] and item['table_footnote'] == []:  # without caption and footnote in the block
                    prev_pos, next_pos = 9999, 9999
                    prev_id, next_id = None, None
                    prev_img_ids, next_img_ids = [], []

                    # check next block for image description info
                    if idx < len(self.pdf_json) - 1 and self.pdf_json[idx+1]['type'] == 'text' and self.pdf_json[idx+1].get('text') is not None:
                        next_item = self.pdf_json[idx+1]
                        mtch_rslts = re.finditer(IMG_REGX_NAME_PTRN, next_item['text'], re.IGNORECASE)

                        next_tbl_ids = []
                        for match in mtch_rslts:
                            next_tbl_ids.append(match.group(0)) 

                        for id in next_tbl_ids:
                            if id not in tbl_id_lst:
                                next_id = id
                                next_pos = next_item['text'].index(id)
                                next_tbl_ids = next_tbl_ids.remove(id)
                                break

                    # check previous block for image description info
                    if idx > 1 and self.pdf_json[idx-1]['type'] == 'text' and self.pdf_json[idx-1].get('text') is not None:
                        prev_item = self.pdf_json[idx-1]
                        mtch_rslts = re.finditer(IMG_REGX_NAME_PTRN, prev_item['text'], re.IGNORECASE)

                        prev_tbl_ids = []
                        for match in mtch_rslts:
                            prev_tbl_ids.append(match.group(0)) 

                        for id in prev_tbl_ids:
                            if id not in tbl_id_lst:
                                prev_id = id
                                prev_pos = prev_item['text'].index(id)
                                prev_tbl_ids = prev_tbl_ids.remove(id)
                                break
                    
                    if next_pos < prev_pos:
                        item['id'] = next_id
                        item['related_ids'] = next_img_ids
                        item['if_aligned'] = True
                        item['table_caption'] = [next_item['text']]
                        next_item['if_aligned'] = False

                    elif prev_pos < next_pos:
                        item['id'] = prev_id
                        item['related_ids'] = prev_img_ids
                        item['if_aligned'] = True
                        item['table_caption'] = [prev_item['text']]
                        prev_item['if_aligned'] = False
                        
                else:
                    desc = "\n".join(item.get('img_caption', [])) + "\n" + "\n".join(item.get('img_footnote', []))
                    mtch_rslts = re.finditer(IMG_REGX_NAME_PTRN, desc, re.IGNORECASE)

                    tbl_ids = []
                    for match in mtch_rslts:
                        tbl_ids.append(match.group(0)) 
                    
                    for id in tbl_ids:
                        if id not in tbl_id_lst:
                            item['id'] = id
                            item['related_ids'] = tbl_ids.remove(id)
                            item['if_aligned'] = True
                            break
