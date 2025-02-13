

import re 
import os
import sys
import time
import requests
from typing import List, Dict, Optional


import fitz
import toml
import copy
import zipfile
from bs4 import BeautifulSoup


from pdf_process.pdf_meta_det import extract_meta, dump_toml
from pdf_process.pdf_toc_gen import get_file_encoding, gen_toc

SECTION_TITLES = ["Abstract",
                'Introduction', 'Related Work', 'Background',
                "Introduction and Motivation", "Computation Function", " Routing Function",
                "Preliminary", "Problem Formulation",
                'Methods', 'Methodology', "Method", 'Approach', 'Approaches',
                "Materials and Methods", "Experiment Settings",
                'Experiment', "Experimental Results", "Evaluation", "Experiments",
                "Results", 'Findings', 'Data Analysis',
                "Discussion", "Results and Discussion", "Conclusion",
                'References',
                "Acknowledgments", "Appendix", "FAQ", "Frequently Asked Questions"]


def download_file(url, filename):
    """Downloads a file from the given URL and saves it as filename."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        with open(filename, 'wb') as f:
            f.write(response.content)

        print(f"Successfully downloaded: {filename}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading: {e}")

def unzip_file(original_zip_file, destination_folder):
    assert os.path.splitext(original_zip_file)[-1] == '.zip'
    with zipfile.ZipFile(original_zip_file, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)

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

# OUtline Detection
class PDFOutline:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def toc_extraction(self, excpert_len:Optional[int]=300):
        """apply pymupdf to extract outline
        Args:
            pdf_path: path to pdf file
            excpert_len: excerpt lenght of initial text
        Return:
            pdf_toc: pdf toc including level, title, page, position, nameddest, if_collapse, excerpt
                     if_collapse: if contains next level title
                     excerpt: initial text
        """
        doc = fitz.open(self.pdf_path)
        toc_infos = doc.get_toc(simple=False) or []

        pdf_toc = []
        for item in toc_infos:
            lvl = item[0] if len(item) > 0 else None
            title = item[1] if len(item) > 1 else None
            start_page = item[2] if len(item) > 2 else None
            end_pos = item[3].get('to') if len(item) > 3 and item[3] else None
            nameddest = item[3].get('nameddest') if len(item) > 3 and item[3] else None
            if_collapse = item[3].get('collapse', False) if len(item) > 3 and item[3] else None

            if start_page is not None:
                page = doc[start_page-1]
                blocks = page.get_text("blocks")

                lines = ""
                for block in blocks:
                    x0, y0, x1, y1, text, _, _ = block
                    if len(lines) < excpert_len:
                        if end_pos and x0 >= end_pos[0]:
                            lines += text
                    else:
                        break

                pdf_toc.append({
                    "level": lvl,
                    "title": title,
                    "page": start_page,
                    "position": end_pos,
                    "nameddest": nameddest,
                    'if_collapse': if_collapse,
                    "excerpt": lines + "..."
                })
        return pdf_toc
    
    def toc_detection(self, titles=SECTION_TITLES):
        """Code not ready
        requires detction models
        """
        mtch_rslts = []
        try:
            doc = fitz.open(self.pdf_path)
            pattern = '|'.join(re.escape(title) for title in titles)

            for i in range(len(doc)):  # 扫描所有页面
                tmp_rslt = extract_meta(doc, pattern=pattern, page=i + 1)
                mtch_rslts.extend(tmp_rslt)

            # 移除错误的过滤逻辑，直接使用所有匹配结果
            rvsd_mtch_rslts = mtch_rslts

            auto_level = 1
            addnl = False
            tmp_meta_ptrn = [dump_toml(m, auto_level, addnl) for m in rvsd_mtch_rslts]

            # 将 tmp_meta_ptrn 写入 recipe.toml 文件
            with open('recipe.toml', 'w', encoding='utf-8') as f:
                f.write('\n'.join(tmp_meta_ptrn))

            recipe_file_path = 'recipe.toml'
            recipe_file = open(recipe_file_path, "r", encoding=get_file_encoding(recipe_file_path))
            recipe = toml.load(recipe_file)
            toc = gen_toc(doc, recipe)
            return toc

        except Exception as e:
            print(f"处理 PDF 文件时出错: {self.pdf_path}, 错误信息: {e}")
            return None # 或者抛出异常，根据实际需求决定

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
        """将 Markdown 文本中的 Markdown 表格转换为 HTML 表格。
        Args:
            markdown_text: 包含 Markdown 表格的 Markdown 文本。
        Returns:
            转换后的 Markdown 文本，表格部分已转换为 HTML 表格。
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

        return "\n".join(output_lines)


    def align_md_images(self):
        """Converts HTML image tags within text to Markdown image syntax.
        Args:
            html_text: The input text containing HTML image tags.
        Returns:
            The text with HTML image tags converted to Markdown image syntax.
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

        return regex.sub(replace_image, self.pdf_md)


    def align_md_toc(self):
        """
        Align markdown title with pdf table of content (generated from fitz)

        Args:
            md_file: Path to the markdown file.
            pdf_toc: pdf toc from pdf_outline_detection function

        Returns:
            A list of dictionaries, where each dictionary represents a section
            with 'level', 'section_num', 'title', and 'text' keys.
            Returns an empty list if the file doesn't exist.
            Returns None if an error occurs.
        """
        if self.pdf_toc:
            modified_lines = []  # 用于存储修改后的行的列表

            title_pattern = r"^#{1,}\s*.*$"  # patttern of markdown title
            md_titles = []

            for idx, line in enumerate(self.pdf_md.splitlines()):  # iterate markdown lines
                if line.strip() not in ["\n", "\s", "\r", ""]:
                    match = re.search(title_pattern, line)
                    if match:  # find markdown title
                        sec_title = line
                        flag = 0

                        for x in self.pdf_toc:  # iterate pdf toc, refine markdown title based on toc title
                            toc_title = x['title'] 
                            toc_level = int(x['level'])  
                            if toc_title in line:  
                                sec_title = "#"*toc_level + " " + toc_title + "  "
                                flag = 1
                                break
                        
                        if flag == 0:  # markdown title not exit in toc
                            for item in ['Acknowledgement', 'Reference', 'Appendix']:
                                if item in line:
                                    sec_title = line
                                    flag = 1
                        
                        if flag == 0:
                            if len(md_titles) > 0:
                                if re.match('^#{1,}', md_titles[-1]):
                                    pre_level = re.match('^#{1,}', md_titles[-1]).group(0) + "#"
                                    sec_title = re.sub('^#{1,}', pre_level, line)
                                else:
                                    sec_title = "#" + line

                        modified_lines.append(sec_title)
                        md_titles.append(sec_title)  # get markdown title

                    else:
                        modified_lines.append(line)
        return "\n".join(modified_lines), md_titles


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
        img_lst_rvsd = copy.deepcopy(img_lst)
        
        for idx, line in enumerate(lines):
            if line.strip() not in ["\n", "\s", "\r", ""]:

                # image match logic
                img_matches = list(re.finditer(img_ptrn, line))  # 使用 finditer 获取所有匹配项

                if img_matches:
                    for match in reversed(img_matches):  # 逆序遍历匹配项，避免替换位置错乱
                        alt_text = match.group(1).strip()
                        image_url = match.group(2)
                        title = match.group(4).strip() if match.group(4) else None

                        for item in img_lst_rvsd:
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
        return "\n".join(lines), img_lst_rvsd
    
    
    def modify_tables_info(self, tbl_lst):
        """update table information with alternative text, image title, etc."""
        
        tbl_lst_rvsd = copy.deepcopy(tbl_lst)

        lines = self.pdf_md.splitlines()

        for idx, line in enumerate(lines):  # iterate lines
            soup = BeautifulSoup(line, 'html.parser')
            table = soup.find('table')

            if table:
                for item in tbl_lst_rvsd:  # iterate over table list 
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

        return "\n".join(lines), tbl_lst_rvsd



class MarkdownSeg:
    def __init__(self, md_content):
        self.md_content = md_content

    def md_seg_by_title(self, level):
        title_pattern = re.compile(rf"^#{{{level}}}\s+(.+)$", re.MULTILINE)

        segments = []

        lines = []
        current_section = ""
        current_title = ""

        num = 1  # Initialize section number
        para_id = 1  # initialize pragraph number

        for idx, line in enumerate(self.md_content.splitlines()):
            if line.strip() not in ["\n", "\s", "\r", ""] and len(line) < 100:
                match = title_pattern.match(line)
                if match:
                    if current_section:  # Save the previous section
                        segments.append({
                            'level': level,
                            'num': num,
                            'title': current_title,
                            'text': current_section.strip(),  # Remove leading/trailing whitespace
                            'lines': lines
                        })
                        num += 1  # Increment for the next section
                    
                    # ready for next section
                    current_title = match.group(1).strip()
                    current_section = ""  # Start a new section (no title line)
                    lines = []
                    para_id = 1
                else:
                    current_section += line + "\n"  # Add to the current section
                    lines.append({'id': idx, 'line': line})
                    para_id += 1

        if current_section:  # Save the last section
            segments.append({
                'level': level,
                'num': num,
                'title': current_title,
                'text': current_section.strip(),
                'lines': lines
            })

        return segments

    def restore_seg_information(md_text, img_lst, tbl_lst, ref_lst):
        """restore images, tables, references within md_text
        
        """
        lines = md_text.splitlines()

        seg_images, seg_tbls, seg_refs = [], [], []
        for idx, line in enumerate(lines):
            if line.strip() not in ["\n", "\s", "\r", ""]:
                # resore images in segment
                for img in img_lst:
                    md_ref = img.get('mod_md_ref', '').strip()
                    # image cited in line but not exist in section 
                    if (md_ref not in "\n".join(lines).strip()
                        and (img.get('id') in line.strip() or img.get('title') in line.strip())):
                        lines.insert(idx+1, md_ref)
                        if img not in seg_images:
                            seg_images.append(img)

                    # line contains image ref but not cited in section
                    if md_ref in line.strip():
                        if img.get('id') not in "\n".join(lines).strip() or img.get('title') in "\n".join(lines).strip():
                            lines[idx] = line.replace(md_ref, "  ")
                        elif img not in seg_images:
                            seg_images.append(img)

                # resore tables in segment
                for tbl in tbl_lst:
                    md_ref = tbl.get('mod_md_ref').strip()

                    # image cited in line but not exist in section 
                    if (md_ref not in "\n".join(lines).strip()
                        and (tbl.get('id') in line.strip() or tbl.get('title') in line.strip())):
                        lines.insert(idx+1, md_ref)
                        if tbl not in seg_tbls:
                            seg_tbls.append(tbl)

                    # line contains image ref but not cited in section
                    if md_ref in line.strip():
                        if (tbl.get('id') not in "\n".join(lines).strip() or tbl.get('title') in "\n".join(lines).strip()):
                            lines[idx] = line.replace(md_ref, "  ")
                        elif tbl not in seg_tbls:
                            seg_tbls.append(tbl)     

                # resore refs in segment
                for idx, line in enumerate(lines):
                    if line.strip() not in ["\n", "\s", "\r", ""]:
                        for ref in ref_lst:
                            if ref not in seg_refs:
                                contexts = ref.get('contexts')
                                for x in contexts:
                                    if x.strip() in line:
                                        seg_refs.append(ref.get('citedPaper', {}))  # get only ref paper information, neglect isInfluential, intent, etc.
                                        break
        
        # to-do: append references here
        # if len(seg_refs) > 0:
        #     lines.extend()

        return "\n".join(lines), seg_images, seg_tbls, seg_refs