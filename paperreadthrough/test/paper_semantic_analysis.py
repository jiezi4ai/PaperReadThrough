import os
import json
import time
import PIL.Image

from typing import List, Dict, Optional

from models.default_models import llm_gen_w_retry, llm_image_gen_w_retry
from prompts.semantic_analysis_prompts import *

class PaperComprehension:
    def __init__(self, final_pdf_json, data_path):
        self.pdf_processed_json = final_pdf_json
        self.data_path = data_path

    
    def paper_topic_analysis(self, api_key, model_name, domain):
        """extract topics from each section"""
        responses = []
        for section in self.processed_json:
            title = "#" * section.get('level') + " " + section.get('title')
            md_text = section.get('refined_text')
            md_lines = "\n".join([f"<line_id> {x.get('id')} <\line_id>  <line_text> {x.get('line')} <\line_text>" for x in section.get('lines')])
            input_text = title + "\n" + md_lines
            
            images = section.get('images')
            if title not in ["References", "Acknowledgments"] and len(md_text) > 200:
                imgs_prompt = ""
                pil_images = []
                if len(images) > 0:
                    img_info = ""
                    for img in images:
                        img_title = img.get('title')
                        img_url = os.path.join(self.data_path, img.get('img_path'))
                        pil_images.append(PIL.Image.open(img_url))
                        img_info += f"- image title: {img_title}  attached image: {os.path.basename(img_url)} \n"
                    imgs_prompt = f"Here are images mentioned in markdown text:\n{img_info}"
                
                    qa_prompt = topics_prompt.format(
                        domain = domain,
                        example_json = json.dumps(topics_example_json, ensure_ascii=False), 
                        markdown_text = input_text,
                        further_information = imgs_prompt)

                    res = llm_image_gen_w_retry(
                        api_key=api_key, model_name='gemini-2.0-flash-thinking-exp', 
                        qa_prompt=qa_prompt, pil_images=pil_images, sys_prompt=None, temperature=0.6)

                else:
                    qa_prompt = topics_prompt.format(
                        domain = domain,
                        example_json = json.dumps(topics_example_json, ensure_ascii=False), 
                        markdown_text = input_text,
                        further_information = "")

                    res = llm_gen_w_retry(
                        api_key=api_key, model_name=model_name, 
                        qa_prompt=qa_prompt, sys_prompt=None, temperature=0.6)
                responses.append(res)
                time.sleep(5)
        return responses
    
    def paper_keywords_analysis(self, api_key, model_name, domain):
        """extract addtional information for segments
        """
        addtional_infos = []
        for section in self.pdf_processed_json:
            title = "#" * section.get('level') + " " + section.get('title')
            md_text = section.get('refined_text')
            input_text = title + "\n" + md_text
            
            if title not in ["References", "Acknowledgments"] and len(md_text) > 200:
                qa_prompt = additional_info_prompt.format(
                    domain = domain,
                    example_json = json.dumps(additional_example_json, ensure_ascii=False),
                    markdown_text = input_text)

                res = llm_gen_w_retry(
                    api_key=api_key, model_name=model_name, 
                    qa_prompt=qa_prompt, sys_prompt=None, temperature=0.6)
                addtional_infos.append(res)
                time.sleep(5)
        return addtional_infos
    
    def paper_conclusion_analysis(self, abstract, topics_lst, api_key, model_name, domain):
        abs_md_text = abstract

        intro_md_text, met_text, con_md_text = "", "", ""
        for item in self.pdf_processed_json:
            title = item.get('title').strip()
            md_text = item.get('refined_text')
            if title.lower() in ['introduction', 'overview']:
                intro_md_text = md_text
            elif title.lower() in ['method', 'methodology', 'approach', 'framework']:
                met_text = md_text
            elif title.lower() in ['conclusion', 'summary']:
                con_md_text = md_text


        sum_md_text = """# Key Information  
        {md_text}
        """.format(md_text="\n".join(topics_lst))