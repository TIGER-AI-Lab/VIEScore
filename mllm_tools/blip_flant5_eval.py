import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from typing import List
import torch
from io import BytesIO
from mllm_tools.utils import merge_images

class BLIP_FLANT5():
    def __init__(self, model_id:str="Salesforce/blip2-flan-t5-xxl") -> None:
        """
        BLIP_FLANT5 tends to otuput shorter text, like "a tiger and a zebra". Try to design the prompt with shorter answer.
        requires: pip install accelerate transformers==4.35.2
        Args:
            model_id (str): BLIP_FLANT5 model name, e.g. "Salesforce/blip2-flan-t5-xxl"
        """
        self.model_id = model_id
        self.processor = Blip2Processor.from_pretrained(model_id)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
        
    def prepare_prompt(self, image_links: List = [], text_prompt: str = ""):
        if type(image_links) == str:
            image_links = [image_links]
        image = merge_images(image_links)
        inputs = self.processor(image, text_prompt, return_tensors="pt").to(self.model.device)
        return inputs

    def get_parsed_output(self, inputs):
        generation_output = self.model.generate(**inputs, max_new_tokens=512)
        generation_text = self.processor.batch_decode(generation_output, skip_special_tokens=True)
        return generation_text[0].strip(" \n")
    

if __name__ == "__main__":
    blip = BLIP_FLANT5("Salesforce/blip2-flan-t5-xxl")
    prompt = blip.prepare_prompt(['../Walking_tiger_female.jpg', 'https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg'], 'What is difference between two images?')
    print(blip.get_parsed_output(prompt))
    """
    Output: a tiger and a zebra
    """