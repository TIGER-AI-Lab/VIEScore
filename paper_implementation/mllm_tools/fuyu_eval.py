import requests
import torch
from transformers import FuyuProcessor, FuyuForCausalLM, AutoTokenizer
from PIL import Image
from typing import List
from io import BytesIO
from mllm_tools.utils import merge_images

class Fuyu():
    def __init__(self, model_id:str="adept/fuyu-8b") -> None:
        """
        requires: pip install transformers==4.35.2
        Args:
            model_id (str): Fuyu model name, e.g. "adept/fuyu-8b"
        """
        self.model_id = model_id
        self.processor = FuyuProcessor.from_pretrained(model_id)
        self.model = FuyuForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
    
    def prepare_prompt(self, image_links: List = [], text_prompt: str = ""):
        if type(image_links) == str:
            image_links = [image_links]
        image = merge_images(image_links)
        inputs = self.processor(text=text_prompt, images=image, return_tensors="pt").to(self.model.device)
        return inputs

    def get_parsed_output(self, inputs):
        generation_output = self.model.generate(**inputs, max_new_tokens=512, pad_token_id=self.pad_token_id)
        input_len = inputs.input_ids.shape[1]
        generation_text = self.processor.batch_decode(generation_output[:, input_len:], skip_special_tokens=True)
        return generation_text[0].strip(" \n")
    
if __name__ == "__main__":
    # Tend to output nothing in practice
    fuyu = Fuyu("adept/fuyu-8b")
    prompt = fuyu.prepare_prompt(['../Walking_tiger_female.jpg', 'https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg'], 'What is difference between two images?')
    print(fuyu.get_parsed_output(prompt))
    """
    Output: In the image on the left, a tiger is walking through a grassy field. In the image on the right, a zebra is walking through a grassy field.
    """