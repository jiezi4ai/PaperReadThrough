# PDF Markdown Segmentation
# Run pdf_outline_gen, minerU process, pdf_post_process before this

import re 
from typing import List, Dict, Optional

class MdSeg:
    def __init__(self, md_content):
        self.md_content = md_content

    def md_seg_by_title(self, level):
        title_pattern = re.compile(rf"^#{{{level}}}\s+(.+)$", re.MULTILINE)
        lines = self.md_content.splitlines()

        segments = []
        current_section = ""
        current_title = ""

        num = 1  # Initialize section number
        para_id = 1  # initialize pragraph number
        for line in lines:
            if line.strip() not in ["\n", "\s", "\r", ""] and len(line) < 100:
                match = title_pattern.match(line)
                if match:
                    if current_section:  # Save the previous section
                        segments.append({
                            'level': level,
                            'num': num,
                            'title': current_title,
                            'text': current_section.strip(),  # Remove leading/trailing whitespace
                        })
                        num += 1  # Increment for the next section
                    
                    # ready for next section
                    current_title = match.group(1).strip()
                    current_section = ""  # Start a new section (no title line)
                    para_id = 1
                else:
                    current_section += line + "\n"  # Add to the current section
                    para_id += 1

        if current_section:  # Save the last section
            segments.append({
                'level': level,
                'num': num,
                'title': current_title,
                'text': current_section.strip()
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