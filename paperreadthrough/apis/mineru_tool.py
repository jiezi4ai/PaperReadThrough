# minerU API from https://mineru.net/apiManage/docs
# Note: 
# 1. recommend batch process for efficiency
# To-do 
# Note: monitor_batch_status need to be further tested
import os
import uuid
import copy
import zipfile
import aiohttp
import asyncio
import requests
from pathlib import Path  
from typing import List, Optional

TASK_URL = "https://mineru.net/api/v4/extract/task"
BATCH_URL = "https://mineru.net/api/v4/file-urls/batch"
BATCH_STATUS_URL = "https://mineru.net/api/v4/extract-results/batch"


def detect_lang(string):
    """
    检查整个字符串是否包含中文
    :param string: 需要检查的字符串
    :return: bool
    """

    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return 'zh'
    return 'en'


def unzip_file(original_zip_file, destination_folder):
    assert os.path.splitext(original_zip_file)[-1] == '.zip'
    with zipfile.ZipFile(original_zip_file, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)
        print(f"Successfully unzipped: {destination_folder}")


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


class MinerUKit:
    def __init__(self, api_key):
        self.api_key = api_key
        self.task_url = TASK_URL
        self.batch_url = BATCH_URL
        self.batch_status_url = BATCH_STATUS_URL
        self.header = {
                    'Content-Type':'application/json',
                    "Authorization":f"Bearer {self.api_key}"
                 }
        self.config = {
            "enable_formula": True,
            "language": "en",
            "layout_model":"doclayout_yolo",
            "enable_table": True
        }


    async def single_process_url_async(self, pdf_url, if_ocr, lang):
        """apply MinerU API to process single PDF asynchronously
        """
        data = copy.deepcopy(self.config)
        data['url'] = pdf_url
        data['is_ocr'] = if_ocr
        data['language'] = lang
        async with aiohttp.ClientSession() as session: # 使用 aiohttp.ClientSession()
            async with session.post(url=self.task_url, headers=self.header, json=data) as response: # 使用 session.post
                print(response.status) # aiohttp 中 status 是属性，不是 status_code
                return await response.read() # 使用 await response.read() 获取响应内容，或者 response.json() 获取 JSON 数据


    async def batch_process_files_async(self, pdf_files: List[str], if_ocr: Optional[bool]=False, lang: Optional[str]='en'):
        """apply MinerU API to process multiple PDF in local path asynchronously
        """
        files = []
        for file in pdf_files:
            files.append({"name": os.path.basename(file),
                        "data_id": str(uuid.uuid1())})
        data = copy.deepcopy(self.config)
        data['is_ocr'] = if_ocr
        data['language'] = lang
        data['files'] = files

        try:
            async with aiohttp.ClientSession() as session: # 使用 aiohttp.ClientSession()
                async with session.post(url=self.batch_url, headers=self.header, json=data) as response: # 使用 session.post
                    if response.status == 200: # aiohttp 中 status 是属性，不是 status_code
                        result = await response.json() # 使用 await response.json() 获取 JSON 数据
                        print('response success. result:{}'.format(result))
                        if result["code"] == 0:
                            batch_id = result["data"]["batch_id"]
                            urls = result["data"]["file_urls"]
                            print('batch_id:{},urls:{}'.format(batch_id, urls))

                            for idx, file_path in enumerate(pdf_files):
                                with open(file_path, 'rb') as f:
                                    async with session.put(urls[idx], data=f) as res_upload: # 使用 session.put
                                        if res_upload.status == 200: # aiohttp 中 status 是属性，不是 status_code
                                            print("upload success")
                                        else:
                                            print("upload failed")
                        else:
                            print('apply upload url failed,reason:{}'.format(result.get("msg"))) # 使用 result.get("msg") 避免 KeyError
                    else:
                        print('response not success. status:{} ,result:{}'.format(response.status, response)) # aiohttp 中 status 是属性，不是 status_code
                    return response # 返回 aiohttp 的 response 对象
        except Exception as err:
            print(err)

        return None


    async def batch_process_urls_async(self, pdf_urls: List[str], if_ocr: Optional[bool]=False, lang: Optional[str]='en'):
        """apply MinerU API to process multiple PDF urls asynchronously
        """
        files = []
        for pdf_url in pdf_urls:
            files.append({"url": pdf_url,
                        "data_id": str(uuid.uuid1())})
        data = copy.deepcopy(self.config)
        data['is_ocr'] = if_ocr
        data['language'] = lang
        data['files'] = files

        try:
            async with aiohttp.ClientSession() as session: # 使用 aiohttp.ClientSession()
                async with session.post(url=self.batch_url, headers=self.header, json=data) as response: # 使用 session.post
                    if response.status == 200: # aiohttp 中 status 是属性，不是 status_code
                        result = await response.json() # 使用 await response.json() 获取 JSON 数据
                        print('response success. result:{}'.format(result))
                        if result["code"] == 0:
                            batch_id = result["data"]["batch_id"]
                            print('batch_id:{}'.format(batch_id))
                        else:
                            print('submit task failed,reason:{}'.format(result.get("msg"))) # 使用 result.get("msg") 避免 KeyError
                    else:
                        print('response not success. status:{} ,result:{}'.format(response.status, response)) # aiohttp 中 status 是属性，不是 status_code
                    return response # 返回 aiohttp 的 response 对象
        except Exception as err:
            print(err)

        return None


    async def batch_status_check_async(self, batch_id): # 假设 batch_status_check 已经有异步版本
        """check status code of batch task
        """
        async with aiohttp.ClientSession() as session:
            url = f'{self.batch_status_url}/{batch_id}'
            headers = self.header
            async with session.get(url, headers=headers) as response:
                return response 
            
    
    def download_and_unzip(self, zip_url, download_file_name, unzip_folder_name):
        """download and unzip MinerU processed files"""
        download_file(zip_url, download_file_name)
        unzip_file(download_file_name, unzip_folder_name)
        os.remove(download_file_name) 

        for file in Path(unzip_folder_name).glob('*'): 
            file_nm = os.path.basename(file)
            if "_origin.pdf" in file_nm:
                os.remove(file) 
            elif "_content_list.json" in file_nm:
                os.rename(file, os.path.join(unzip_folder_name, "content_list.json"))


    async def monitor_batch_status_async(self, batch_id, save_path, interval=10, max_retries=10):
        """
        异步监控批次运行状态，尝试下载，最大重试次数
        Args:
            batch_id: batch id
            save_path: 保存处理后文件的路径 (文件夹名称与原始 pdf 对齐)
            interval: 下次检查的时间间隔 (秒)
            max_retries: 最大重试次数
        Note:
            处理后的数据将保存到文件夹中，文件夹名称与原始 pdf 对齐。
            文件包括:
                - full.md: 最终 markdown 文件
                - _content_list.json: 段落信息
                - layout.json: 详细位置等。
        """
        downloaded_files = set()  # 记录已下载的文件名，避免重复下载

        for _ in range(max_retries):
            running_res = await self.batch_status_check_async(batch_id) # 使用异步的 batch_status_check_async
            if (await running_res.json()).get('msg') == 'ok': # await 获取 json 内容
                results = (await running_res.json()).get('data', {}).get('extract_result', []) # await 获取 json 内容
                for item in results:
                    if item.get('state') == 'done':
                        file_name = item.get('file_name')
                        if file_name not in downloaded_files:  # 检查是否已下载
                            file_name_nosuffix = file_name.rsplit('.', 1)[0]
                            zip_url = item.get('full_zip_url')
                            download_file_name = os.path.join(save_path, file_name_nosuffix + ".zip")
                            unzip_folder_name = os.path.join(save_path, file_name_nosuffix)

                            # 使用 asyncio.to_thread 异步执行 CPU 密集型或同步 I/O 操作，避免阻塞事件循环
                            await asyncio.to_thread(
                                self.download_and_unzip,
                                zip_url, download_file_name, unzip_folder_name
                            )

                            downloaded_files.add(file_name)  # 标记为已下载

                # 检查是否全部完成
                all_done = all(item.get('state') == 'done' for item in results)
                if all_done:
                    print(f"Batch {batch_id} complete") # 拼写错误修正：complte -> complete
                    return

            print(f"Batch {batch_id} running, recheck in next {interval} seconds...")
            await asyncio.sleep(interval) # 使用 asyncio.sleep 进行非阻塞等待

        print(f"Exit as batch {batch_id} reached max retries.")
