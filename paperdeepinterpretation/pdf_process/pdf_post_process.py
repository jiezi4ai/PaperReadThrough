# paper pdf post process
# ideally the paper table of content (toc), paper markdown text, paper content list of json are ready
# refer to mineru_tool.py for pdf processing 
import re 
import os
import requests
from typing import List, Dict, Optional

import copy
import string
import zipfile
from bs4 import BeautifulSoup
from thefuzz import fuzz # pip install thefuzz  https://github.com/seatgeek/thefuzz

from pdf_process import APPENDDIX_TITLES


def remove_non_text_chars(text):
    """remove non text chars
    """
    valid_chars = string.ascii_letters + string.digits  # 包含所有字母和数字的字符串
    cleaned_text = ''
    for char in text:
        if char in valid_chars:
            cleaned_text += char
    return cleaned_text


def get_first_lines(text, sentence_length):
    """"get first line of text"""
    if not text:
        return ""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|;|!)\s', text) # 更精确的断句正则

    result = ""
    current_length = 0
    for sentence in sentences:
        cleaned_sentence = sentence.strip()
        if cleaned_sentence:
            result += cleaned_sentence + " "
            current_length = len(result.strip())

            if current_length >= sentence_length:
                return result.strip()
    return result.strip()


def _convert_table_lines_to_html(table_lines):
    """将 Markdown 表格行转换为 HTML 表格。
    Args:
        table_lines: Markdown 表格行的列表。
    Returns:
        HTML 表格字符串。
    """
    html_lines = ["<table>", "  <thead>", "    <tr>"]
    header_cells = [cell.strip() for cell in table_lines[0].strip('|').split('|')]
    for header in header_cells:
        html_lines.append(f"      <th>{header}</th>")
    html_lines.append("    </tr>")
    html_lines.append("  </thead>")
    html_lines.append("  <tbody>")

    if len(table_lines) > 1 and re.match(r'^\|[-:| ]+\|[-:| ]*$', table_lines[1].strip()):
        # 存在分隔行，跳过分隔行，从第三行开始是数据行
        data_start_index = 2
    else:
        data_start_index = 1 # 没有分隔行，从第二行开始是数据行

    for i in range(data_start_index, len(table_lines)):
        html_lines.append("    <tr>")
        data_cells = [cell.strip() for cell in table_lines[i].strip('|').split('|')]
        for cell in data_cells:
            html_lines.append(f"      <td>{cell}</td>")
        html_lines.append("    </tr>")

    html_lines.append("  </tbody>")
    html_lines.append("</table>")
    return "\n".join(html_lines)


class PDFProcess:
    def __init__(self, pdf_path, pdf_toc, pdf_md, pdf_json):
        """load pdf related files and data
        Args:
            pdf_path: path to pdf file
            pdf_toc: table of content genreated from PDFOutline class
            pdf_md: markdown generated from MinerU after processing PDF ("full.md")
            pdf_json: json content from MinerU after processing PDF ("_content_list.json")
        """
        self.pdf_path = pdf_path
        self.pdf_toc = pdf_toc
        self.pdf_md = pdf_md
        self.pdf_json = pdf_json


    def alighn_md_tables(self):
        """covert tables in markdown syntax to html form
        """
        lines = self.pdf_md.splitlines()
        output_lines = []
        in_table = False
        table_lines = []

        for line in lines:
            if line.strip().startswith('|'):
                in_table = True
                table_lines.append(line)
            else:
                if in_table:
                    # 表格结束，处理之前收集的表格行
                    html_table = _convert_table_lines_to_html(table_lines)
                    output_lines.append(html_table)
                    in_table = False
                    table_lines = []
                output_lines.append(line)

        # 处理文本末尾可能存在的表格
        if in_table:
            html_table = _convert_table_lines_to_html(table_lines)
            output_lines.append(html_table)

        self.pdf_md = "\n".join(output_lines)


    def align_md_images(self):
        """covert HTML image tags within text to Markdown image syntax.
        """
        def replace_image(match):
            """
            Replaces a single HTML image tag with Markdown image syntax.
            """
            src = match.group('src')
            alt = match.group('alt') or ''  # Default to empty string if alt is missing
            title = match.group('title') or '' # Default to empty string if title is missing

            markdown_image = f"![{alt}]({src}"
            if title:
                markdown_image += f' "{title}"'
            markdown_image += ")"
            return markdown_image

        # Regex to find HTML image tags and capture src, alt, and title attributes
        # It's designed to be relatively robust to attribute order and presence.
        regex = re.compile(
            r'<img.*?src=["\'](?P<src>.*?)["\'].*?(?:alt=["\'](?P<alt>.*?)["\'])?.*?(?:title=["\'](?P<title>.*?)["\'])?.*?/>',
            re.IGNORECASE  # Case-insensitive matching for HTML tags
        )
        self.md = regex.sub(replace_image, self.pdf_md)


    def align_md_toc(self):
        """Align markdown title with pdf table of content 
        Args:
            md_file: Path to the markdown file.
            pdf_toc: pdf toc from pdf_outline_detection function
        Returns:
            A list of dictionaries, where each dictionary represents a section
            with 'level', 'section_num', 'title', and 'text' keys.
            Returns an empty list if the file doesn't exist.
            Returns None if an error occurs.
        """
        lines = self.pdf_md.splitlines()

        modified_lines = [] 
        md_titles_info = []  # store title after modification
        title_pattern = r"^#{1,}\s*.*$"  # patttern of markdown title
        
        for line in lines: 
            new_line = line
            if line.strip() not in ["\n", "\s", "\r", ""] and len(line) < 100:
                ptrn_match = re.match(title_pattern, line)
                if ptrn_match:  # find markdown title
                    flag = 0

                    for toc in self.pdf_toc:  # iterate pdf toc, refine markdown title based on toc title
                        toc_title = toc['title'] 
                        toc_level = int(toc['level'])  
                        if_appendix = toc['if_appendix']
                        if re.search(re.escape(toc_title), line, re.IGNORECASE): # if toc_title in line: 
                            line = "#"*toc_level + " " + toc_title + "  "
                            title_info = {'title': line, 'level': toc_level, 'if_appendix': if_appendix, 'if_modified': True}
                            flag = 1
                            break
                    
                    if flag == 0:  
                        # for appendix
                        pattern = '|'.join(re.escape(title) for title in APPENDDIX_TITLES) 
                        mtch = re.search(pattern, line, re.IGNORECASE)
                        if mtch:
                            level = re.match('^#{1,}', line).group(0).count("#")
                            title_info = {'title': line, 'level': level, 'if_appendix': True, 'if_modified': False}
                            flag = 1
                    
                    if flag == 0:
                        # for others, downgrade one more level
                        if len(md_titles_info) > 0:
                            level = line.count("#") + 1
                            if_appendix = md_titles_info[-1].get('if_appendix')
                            line = re.sub('^#{1,}', '#'*level, line)
                            title_info = {'title': line, 'level': level, 'if_appendix': if_appendix, 'if_modified': True}
                        else:
                            title_info = {'title': line, 'level': 1, 'if_appendix': False, 'if_modified': False}

                    if title_info not in md_titles_info:
                        md_titles_info.append(title_info)  # get markdown title
                    
            modified_lines.append(line)
        self.pdf_md = "\n".join(modified_lines)
        return md_titles_info

    def align_content_json(self):
        """assign title and ids to images/ charts, tables, and equations
        """
        img_lst, tbl_lst, formula_lst = [], [], []
        i, j, k = 1, 1, 1
        for x in self.pdf_json:
            if x['type'] in ['image']:
                desc = "\n".join(x.get('img_caption', [])) + "\n" + "\n".join(x.get('img_footnote', []))
                ptrn = r"(pic|picture|img|image|chart|figure|fig|table|tbl)\s*([0-9]+(?:\.[0-9]+)?|[0-9]+|[IVXLCDM]+|[a-zA-Z]+)"
                mtch_rslts = re.finditer(ptrn, desc, re.IGNORECASE)

                img_ids = []
                for match in mtch_rslts:
                    img_ids.append(match.group(0))  # 直接获取整个匹配的字符串

                if len(img_ids) == 0:
                    img_ids = [f"Image_Number_{i}"]
                    i += 1
                x['id'] = img_ids[0]
                x['related_ids'] = img_ids[1:]
                x['title'] = get_first_lines(desc, 10)
                x['description'] = desc
                img_lst.append(x)

            elif x['type'] == 'table':
                desc = "\n".join(x.get('table_caption', [])) + "\n" + "\n".join(x.get('table_footnote', []))
                ptrn = r"(tbl|table|chart|figure|fig)\s*([0-9]+(?:\.[0-9]+)?|[0-9]+|[IVXLCDM]+|[a-zA-Z]+)"
                mtch_rslts = re.finditer(ptrn, desc, re.IGNORECASE)

                tbl_ids = []
                for match in mtch_rslts:
                    tbl_ids.append(match.group(0))  # 直接获取整个匹配的字符串

                if len(tbl_ids) == 0:
                    tbl_ids = [f"Table_Number_{j}"]
                    j += 1
                x['id'] = tbl_ids[0]
                x['related_ids'] = tbl_ids[1:]
                x['title'] = get_first_lines(desc, 10)
                x['description'] = desc
                tbl_lst.append(x)

                # for table with image
                if x.get('img_path') is not None:
                    item = {'type':'image', 'img_path': x.get('img_path'), 'img_caption': x.get('table_caption'), 'table_footnote': x.get('table_footnote'), 'page_idx': x.get('page_idx')}
                    desc = "\n".join(item.get('img_caption', [])) + "\n" + "\n".join(item.get('img_footnote', []))
                    ptrn = r"(table|tbl|pic|picture|img|image|chart|figure|fig)\s*([0-9]+(?:\.[0-9]+)?|[0-9]+|[IVXLCDM]+|[a-zA-Z]+)"
                    mtch_rslts = re.finditer(ptrn, desc, re.IGNORECASE)

                    img_ids = []
                    for match in mtch_rslts:
                        img_ids.append(match.group(0))  # 直接获取整个匹配的字符串

                    if len(img_ids) == 0:
                        img_ids = [f"Table_Image_Number_{i}"]
                        i += 1
                    item['id'] = img_ids[0]
                    item['related_ids'] = img_ids[1:]
                    item['title'] = get_first_lines(desc, 10)
                    item['description'] = desc
                    img_lst.append(item)

            elif x['type'] == 'equation':

                desc = x.get('text')
                ptrn = r"(formula|equation|notation|syntax)\s*([0-9]+(?:\.[0-9]+)?|[0-9]+|[IVXLCDM]+|[a-zA-Z]+)"
                mtch_rslts = re.finditer(ptrn, desc, re.IGNORECASE)

                equation_ids = []
                for match in mtch_rslts:
                    equation_ids.append(match.group(0))  # 直接获取整个匹配的字符串

                if len(equation_ids) == 0:
                    equation_ids = [f"Equation_Number_{k}"]
                    k += 1
                x['id'] = equation_ids[0]
                x['related_ids'] = equation_ids[1:]
                x['title'] = equation_ids[0]
                x['description'] = equation_ids[0]
                formula_lst.append(x)

                # for table with image
                if x.get('img_path') is not None:
                    item = {'type':'image', 'img_path': x.get('img_path'), 'img_caption': x.get('img_caption'), 'img_caption': x.get('img_caption'), 'page_idx': x.get('page_idx')}
                    desc = item.get('text')
                    ptrn = r"(formula|equation|notation|syntax)\s*([0-9]+(?:\.[0-9]+)?|[0-9]+|[IVXLCDM]+|[a-zA-Z]+)"
                    mtch_rslts = re.finditer(ptrn, desc, re.IGNORECASE)

                    img_ids = []
                    for match in mtch_rslts:
                        img_ids.append(match.group(0))  # 直接获取整个匹配的字符串

                    if len(img_ids) == 0:
                        img_ids = [f"Equation_Image_Number_{i}"]
                        i += 1
                    x['id'] = img_ids[0]
                    x['related_ids'] = img_ids[1:]
                    x['title'] = equation_ids[0]
                    x['description'] = equation_ids[0]
                    img_lst.append(x)
        return img_lst, tbl_lst, formula_lst 


    def modify_image_info(self, img_lst):
        """update image information with alternative text, image title, etc."""
        img_ptrn = re.compile(
            r'!\s*\[\s*(?P<alt>.*?)\s*\]'  # 匹配 ![alt] alt 部分
            r'\s*\(\s*(?P<link>.*?)\s*'   # 匹配 (link) link 部分
            r'(?:'                         # 非捕获组，处理可选的 title 部分
            r'(?P<quotetitle>"(?P<title_double_quote>.*?)"|'  # 匹配 双引号 title， 命名组 quotetitle 和 title_double_quote
            r"'(?P<title_single_quote>.*?)')"                 # 匹配 单引号 title， 命名组 title_single_quote
            r')?'                          # title 部分可选
            r'\s*\)'                      # 匹配 ) 括号结尾
        )
        lines = self.pdf_md.splitlines()
        
        for idx, line in enumerate(lines):
            if line.strip() not in ["\n", "\s", "\r", ""]:
                # image match logic
                img_matches = list(re.finditer(img_ptrn, line))  # 使用 finditer 获取所有匹配项

                if img_matches:
                    for match in reversed(img_matches):  # 逆序遍历匹配项，避免替换位置错乱
                        alt_text = match.group(1).strip()
                        image_url = match.group(2)
                        title = match.group(4).strip() if match.group(4) else None

                        for item in img_lst:
                            if item.get('img_path') == image_url:
                                alt_text = item.get('description') if alt_text is None or alt_text == "" else alt_text
                                title = item.get('title', "") if title is None or title == "" else title
                                title = f"{item.get('id')}: {title}" if item.get('id').lower() not in title.lower() else title
                                img_md = f"![{alt_text.strip()}]({image_url.strip()} '{title.strip()}')"

                                # 计算替换的起始和结束位置
                                start, end = match.span()
                                if item.get('org_md_ref') is None:
                                    item['org_md_ref'] = line[start:end]  # 在image list中添加原始的markdown引用格式 
                                lines[idx] = line[:start] + img_md + line[end:]  # 精确替换
                                if item.get('mod_md_ref') is None:
                                    item['mod_md_ref'] = line[:start] + img_md + line[end:]  # 在image list中添加修订后的markdown引用格式 

                                # 改进删除重复信息逻辑
                                caption = "\n".join(item.get('img_caption')).strip()
                                footnote = "\n".join(item.get('img_footnote')).strip()

                                # 由于alt_text和title中已经包括了足够的信息，删除上下文中的重复信息
                                if caption and len(caption) > 20 and caption != title:
                                    if idx > 0 and caption in lines[idx-1]:
                                        lines[idx-1] = lines[idx-1].replace(caption, "")
                                    if idx < len(lines) - 1 and caption in lines[idx+1]:
                                        lines[idx+1] = lines[idx+1].replace(caption, "")

                                if footnote and len(footnote) > 20 and footnote != title:
                                    if idx > 0 and footnote in lines[idx-1]:
                                        lines[idx-1] = lines[idx-1].replace(footnote, "")
                                    if idx < len(lines) - 1 and footnote in lines[idx+1]:
                                        lines[idx+1] = lines[idx+1].replace(footnote, "")
                                break  # 找到匹配的 item 后跳出循环
        
        self.pdf_md = "\n".join(lines)
        return img_lst
    
    
    def modify_tables_info(self, tbl_lst):
        """update table information with alternative text, image title, etc."""
        lines = self.pdf_md.splitlines()

        for idx, line in enumerate(lines):  # iterate lines
            soup = BeautifulSoup(line, 'html.parser')
            table = soup.find('table')

            if table:
                for item in tbl_lst:  # iterate over table list 
                    tbl_desc = item.get('description')
                    tbl_caption = "\n".join(item.get('table_caption', [])).strip()
                    tbl_footnote = "\n".join(item.get('table_footnote', [])).strip()
                    tbl_body = BeautifulSoup(item.get('table_body') , 'html.parser').find('table')
                    tbl_title = item.get('title')

                    if table == tbl_body:
                        md_caption = table.find('caption')
                        if md_caption:
                            md_caption.string = tbl_desc      # 将<caption>标签的文本内容替换为 tbl_desc
                        else:
                            # 如果没有<caption>标签，则创建一个新的<caption>标签并添加到table中
                            new_caption_tag = soup.new_tag('caption')
                            new_caption_tag.string = tbl_desc
                            table.insert(0, new_caption_tag) # 将新的<caption>标签插入到table的开头 (作为第一个子元素)
                            
                        lines[idx] = f"<html><body>{table}</body></html>  "
                        # 计算替换的起始和结束位置
                        if item.get('org_md_ref') is None:
                            item['org_md_ref'] = f"<html><body>{tbl_body}</body></html>  " # original table
                        if item.get('mod_md_ref') is None:
                            item['mod_md_ref'] = f"<html><body>{table}</body></html>  "  # table with caption

                        # 由于alt_text和title中已经包括了足够的信息，删除上下文中的重复信息
                        if tbl_caption and len(tbl_caption) > 20 and tbl_caption != tbl_title:
                            if idx > 0 and tbl_caption in lines[idx-1]:
                                lines[idx-1] = lines[idx-1].replace(tbl_caption, "")
                            if idx < len(lines) - 1 and tbl_caption in lines[idx+1]:
                                lines[idx+1] = lines[idx+1].replace(tbl_caption, "")

                        if tbl_footnote and len(tbl_footnote) > 20 and tbl_footnote != tbl_title:
                            if idx > 0 and tbl_footnote in lines[idx-1]:
                                lines[idx-1] = lines[idx-1].replace(tbl_footnote, "")
                            if idx < len(lines) - 1 and tbl_footnote in lines[idx+1]:
                                lines[idx+1] = lines[idx+1].replace(tbl_footnote, "")

                        break  # 找到匹配的 item 后跳出循环

        self.pdf_md = "\n".join(lines)
        return tbl_lst


    def modify_reference_info(self, reference_metadata):
        lines = self.pdf_md.splitlines()

        for ref in reference_metadata:
            title = ref.get('citedPaper', {}).get('title')
            if title:
                for line in lines:
                    if len(line) < 500:
                        if re.search(re.escape(title), line, re.IGNORECASE):
                            ref['org_md_ref'] = line
                        else:
                            ratio = fuzz.partial_ratio(title, line)
                            if ratio > 80:
                                ref['org_md_ref'] = line
                                break
                            else:
                                ptrn = remove_non_text_chars(title) 
                                line_rvsd =  remove_non_text_chars(line)
                                if re.search(re.escape(ptrn), line_rvsd, re.IGNORECASE):
                                    ref['org_md_ref'] = line
        return reference_metadata