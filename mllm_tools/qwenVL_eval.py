import requests
import tempfile
import os
import regex as re
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from PIL import Image
from typing import List
from io import BytesIO
class QwenVL():
    merged_image_files = []
    def __init__(self, model_id:str="Qwen/Qwen-VL-Chat") -> None:
        """
        requires: pip install transformers==4.35.2 transformers_stream_generator torchvision tiktoken chardet matplotlib
        Args:
            model_id (str): Qwen model name, e.g. "Qwen/Qwen-VL-Chat"
        """
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True, bf16=True).eval()
    
    def prepare_prompt(self, image_links: List = [], text_prompt: str = ""):
        if type(image_links) == str:
            image_links = [image_links]
        input_list = []
        for i, image_link in enumerate(image_links):
            input_list.append({'image': image_link})
        input_list.append({'text': text_prompt})
        query = self.tokenizer.from_list_format(input_list)
        return query
    

    def get_parsed_output(self, query):
        response, history = self.model.chat(self.tokenizer, query=query, history=None)
        return response

    def __del__(self):
        for image_file in self.merged_image_files:
            os.remove(image_file)
    
if __name__ == "__main__":
    QwenVL = QwenVL("Qwen/Qwen-VL-Chat")
    prompt = QwenVL.prepare_prompt(['../Walking_tiger_female.jpg', 'https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg'], 'What is difference between two images?')
    print(QwenVL.get_parsed_output(prompt))
    """
    Output: The two images show two different animals in different environments. The first image shows a tiger walking through a grassy field, while the second image shows a zebra grazing in the grass. The tiger and the zebra have different physical characteristics, such as the tiger's stripes and the zebra's stripes, as well as different behavioral characteristics, such as the tiger's solitary hunting style and the zebra's social grazing habits.
    """
