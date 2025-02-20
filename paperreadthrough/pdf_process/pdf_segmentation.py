# PDF Markdown Segmentation
# Run pdf_outline_gen, minerU process, pdf_post_process before this

import re 
from typing import List, Dict, Optional

class PDFSeg:
    def __init__(self, pdf_json):
        self.pdf_json = pdf_json

    def get_toc_hierachy(self):
        """generate ToC tree
        Args:
            pdf_json:
        Returns:
            tree form hierachy of sections
        """
        toc = []
        section_stack = []

        for i, item in enumerate(self.pdf_json):
            if item['type'] == 'title':
                level = item['text_level']
                title = item['text']

                while section_stack and section_stack[-1]['level'] >= level:
                    popped_section = section_stack.pop()
                    popped_section['end_position'] = i - 1
                    if section_stack:
                        section_stack[-1]['subsection'].append(popped_section)
                    else:
                        toc.append(popped_section)

                new_section = {'title': title, 'level': level, 'start_position': i, 'end_position': -1, 'subsection': []}
                section_stack.append(new_section)

        while section_stack:
            popped_section = section_stack.pop()
            popped_section['end_position'] = len(self.pdf_json) - 1
            if section_stack:
                section_stack[-1]['subsection'].append(popped_section)
            else:
                toc.append(popped_section)

        return toc
    
    def gen_segmentation(self, toc_hierachy, seg_text_length:Optional[int]=20000):
        """segment content json based on toc hierachy"""
        pdf_texts = [item.get('text', '') for item in self.pdf_json]

        all_seg_paras = []
        for section in toc_hierachy:
            section_paras = []
            
            start_pos = section['start_position']
            end_pos = section['end_position']
            
            if len(tmp_text) > seg_text_length and section.get('subsection', []) != []:
                # if the section is too long, then breakdown to subsection
                for subsection in section.get('subsection'):
                    sub_start_pos = subsection['start_position']
                    sub_end_pos = subsection['end_position']
                    section_paras.append(self.pdf_json[sub_start_pos:sub_end_pos+1])
                    tmp_text = "\n".join(pdf_texts[start_pos:end_pos+1])
                    print('subsection', subsection, len(tmp_text))
            else:
                section_paras.append(self.pdf_json[start_pos:end_pos+1])
                tmp_text = "\n".join(pdf_texts[start_pos:end_pos+1])
                print('section', section, len(tmp_text))
                    
            all_seg_paras.extend(section_paras)
        return all_seg_paras

    def restore_seg_information(self, seg_paras):
        """restore images, tables, references within segments
        Args:
            seg_paras: PDF content json organized in segments, data from gen_segmentation function
            img_lst, tbl_lst: 
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

                # restore refs in segment
                for idx, line in enumerate(lines):
                    if line.strip() not in ["\n", "\s", "\r", ""]:
                        for ref in ref_lst:
                            if ref not in seg_refs:
                                contexts = ref.get('contexts')
                                for x in contexts:
                                    if x.strip() in line:
                                        seg_refs.append(ref.get('citedPaper', {}))  # get only ref paper information, neglect isInfluential, intent, etc.
                                        break
        
        # append references to segments
        if len(seg_refs) > 0:
            lines.extend([x.get('org_md_ref') for x in ref_lst])

        return "\n".join(lines), seg_images, seg_tbls, seg_refs