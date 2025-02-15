import time
import ollama
import numpy as np

async def ollama_embedding(model, texts :list[str]) -> np.ndarray:
    embed_text = []
    for text in texts:
      data = ollama.embeddings(model=model, prompt=text)
      embed_text.append(data["embedding"])
    return embed_text

from sentence_transformers import util

def semantic_similarity_matrix(vec_x, vec_y):
    # embeds = await ollama_embedding(text_lst)
    cosine_scores = util.pytorch_cos_sim(vec_x, vec_y)  # 计算余弦相似度矩阵，仅计算上三角部分
    return cosine_scores

import os
from google import genai
from google.genai import types

def llm_gen(api_key, model_name, qa_prompt, sys_prompt=None, temperature=0.3):
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=sys_prompt,
        temperature=temperature)
    response = client.models.generate_content(
        model=model_name, 
        contents=qa_prompt,
        config=config)
    return response.text

def llm_image_gen(api_key, model_name, qa_prompt, pil_images, sys_prompt=None, temperature=0.3):
    """q&a with images
    Args:
        pil_images:
            import PIL.Image
            image = PIL.Image.open('/path/to/image.png')
    """

    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        system_instruction=sys_prompt,
        temperature=temperature)

    response = client.models.generate_content(
        model=model_name,  #　"gemini-2.0-flash-exp",
        contents=[qa_prompt]+pil_images,
        config=config)

    return response.text

def llm_gen_w_retry(api_key, model_name, qa_prompt, sys_prompt=None, temperature=0.3, max_retries=3, initial_delay=1):
    """
    Wraps the llm_gen_w_images function to enable retries on RESOURCE_EXHAUSTED errors.

    Args:
        api_key: API key for the LLM service.
        model_name: Name of the LLM model to use.
        qa_prompt: Question and answer prompt for the LLM.
        pil_images: List of PIL.Image objects.
        temperature: Temperature for LLM response generation.
        max_retries: Maximum number of retries in case of error.
        initial_delay: Initial delay in seconds before the first retry.

    Returns:
        str: The text response from the LLM, or None if max retries are exceeded and still error.
    """
    retries = 0
    delay = initial_delay

    while retries <= max_retries:
        try:
            return llm_gen(api_key, model_name, qa_prompt, sys_prompt, temperature)
        except Exception as e:
            if e.code == 429:
                if retries < max_retries:
                    retries += 1
                    print(f"Rate limit exceeded. Retrying in {delay} seconds (Retry {retries}/{max_retries})...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff for delay
                else:
                    print(f"Max retries reached.  Raising the last exception.")
                    return None # raise  # Re-raise the last exception if max retries are exhausted
            else:
                print(f"Error Code: {e.code} Error Message: {e.message}")
                return None
                # raise  # Re-raise other ClientErrors (not related to resource exhaustion)

    return None # Should not reach here in normal cases as exception is re-raised or value is returned in try block

def llm_image_gen_w_retry(api_key, model_name, qa_prompt, pil_images, sys_prompt=None, temperature=0.3, max_retries=3, initial_delay=1):
    """
    Wraps the llm_gen_w_images function to enable retries on RESOURCE_EXHAUSTED errors.

    Args:
        api_key: API key for the LLM service.
        model_name: Name of the LLM model to use.
        qa_prompt: Question and answer prompt for the LLM.
        pil_images: List of PIL.Image objects.
        sys_prompt: Optional system prompt for the LLM.
        temperature: Temperature for LLM response generation.
        max_retries: Maximum number of retries in case of error.
        initial_delay: Initial delay in seconds before the first retry.

    Returns:
        str: The text response from the LLM, or None if max retries are exceeded and still error.
    """
    retries = 0
    delay = initial_delay

    while retries <= max_retries:
        try:
            return llm_image_gen(api_key, model_name, qa_prompt, pil_images, sys_prompt, temperature)
        except Exception as e:
            if e.code == 429:
                if retries < max_retries:
                    retries += 1
                    print(f"Rate limit exceeded. Retrying in {delay} seconds (Retry {retries}/{max_retries})...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff for delay
                else:
                    print(f"Max retries reached.  Raising the last exception.")
                    return None # raise  # Re-raise the last exception if max retries are exhausted
            else:
                print(f"Error Code: {e.code} Error Message: {e.message}")
                return None
                # raise  # Re-raise other ClientErrors (not related to resource exhaustion)

    return None # Should not reach here in normal cases as exception is re-raised or value is returned in try block


# DeepSeek
# deepseek-chat 模型已全面升级为 DeepSeek-V3，接口不变。 通过指定 model='deepseek-chat' 即可调用 DeepSeek-V3。
# deepseek-reasoner 是 DeepSeek 最新推出的推理模型 DeepSeek-R1。通过指定 model='deepseek-reasoner'，即可调用 DeepSeek-R1。
# 如未指定 max_tokens，默认最大输出长度为 4K。请调整 max_tokens 以支持更长的输出。
# temperature 参数默认为 1.0。

# 我们建议您根据如下表格，按使用场景设置 temperature。
# 场景	温度
# 代码生成/数学解题   	0.0
# 数据抽取/分析	1.0
# 通用对话	1.3
# 翻译	1.3
# 创意类写作/诗歌创作	1.5
from openai import OpenAI

client = OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)


# json output
import json
from openai import OpenAI

client = OpenAI(
    api_key="<your api key>",
    base_url="https://api.deepseek.com",
)

system_prompt = """
The user will provide some exam text. Please parse the "question" and "answer" and output them in JSON format. 

EXAMPLE INPUT: 
Which is the highest mountain in the world? Mount Everest.

EXAMPLE JSON OUTPUT:
{
    "question": "Which is the highest mountain in the world?",
    "answer": "Mount Everest"
}
"""

user_prompt = "Which is the longest river in the world? The Nile River."

messages = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    response_format={
        'type': 'json_object'
    }
)

print(json.loads(response.choices[0].message.content))

# zhipu
# GLM-4-Flash	免费调用：智谱AI首个免费API，零成本调用大模型	128K	4K
# GLM-4V-Flash	免费模型：专注于高效的单一图像理解，适用于图像解析的场景	8K	-
# CogView-3-Flash	免费模型：免费的图像生成模型	1K	支持多分辨率
from zhipuai import ZhipuAI
client = ZhipuAI(api_key="")  # 请填写您自己的APIKey
response = client.chat.completions.create(
    model="glm-4-plus",  # 请填写您要调用的模型名称
    messages=[
        {"role": "system", "content": "你是一个乐于回答各种问题的小助手，你的任务是提供专业、准确、有洞察力的建议。"},
        {"role": "user", "content": "我对太阳系的行星非常感兴趣，尤其是土星。请提供关于土星的基本信息，包括它的大小、组成、环系统以及任何独特的天文现象。"},
    ],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta)


from openai import OpenAI 

client = OpenAI(
    api_key="your api key",
    base_url="https://open.bigmodel.cn/api/paas/v4/"
) 

from openai import OpenAI 

client = OpenAI(
    api_key="your zhipuai api key",
    base_url="https://open.bigmodel.cn/api/paas/v4/"
) 

completion = client.chat.completions.create(
    model="glm-4",  
    messages=[    
        {"role": "system", "content": "你是一个聪明且富有创造力的小说作家"},    
        {"role": "user", "content": "请你作为童话故事大王，写一篇短篇童话故事，故事的主题是要永远保持一颗善良的心，要能够激发儿童的学习兴趣和想象力，同时也能够帮助儿童更好地理解和接受故事中所蕴含的道理和价值观。"} 
    ],
    top_p=0.7,
    temperature=0.9
 ) 
 
 print(completion.choices[0].message)

# 图片
import base64
from zhipuai import ZhipuAI

img_path = "/Users/YourCompluter/xxxx.jpeg"
with open(img_path, 'rb') as img_file:
    img_base = base64.b64encode(img_file.read()).decode('utf-8')

client = ZhipuAI(api_key="YOUR API KEY") # 填写您自己的APIKey
response = client.chat.completions.create(
    model="glm-4v-plus",  # 填写需要调用的模型名称
    messages=[
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
                "url": img_base
            }
          },
          {
            "type": "text",
            "text": "请描述这个图片"
          }
        ]
      }
    ]
)
print(response.choices[0].message)