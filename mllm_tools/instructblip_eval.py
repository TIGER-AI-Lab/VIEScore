import requests
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from typing import List
import torch
from typing import List
from io import BytesIO
from mllm_tools.utils import merge_images

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

class INSTRUCTBLIP_FLANT5():
    def __init__(self, model_id:str="Salesforce/instructblip-flan-t5-xxl") -> None:
        """
        requires: pip install transformers==4.35.2
        Args:
            model_id (str): BLIP_FLANT5 model name, e.g. "Salesforce/blip2-flan-t5-xxl"
        """
        self.model_id = model_id
        self.processor = InstructBlipProcessor.from_pretrained(model_id)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
    
    def prepare_prompt(self, image_links: List = [], text_prompt: str = ""):
        if type(image_links) == str:
            image_links = [image_links]
        image = merge_images(image_links)
        inputs = self.processor(image, text_prompt, return_tensors="pt").to(self.model.device)
        return inputs

    def get_parsed_output(self, inputs):
        generation_output = self.model.generate(**inputs, 
            do_sample=True,
            # num_beams=5,
            max_new_tokens=512,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
        )
        generation_text = self.processor.batch_decode(generation_output, skip_special_tokens=True)
        return generation_text[0].strip(" \n")
    

if __name__ == "__main__":
    blip = INSTRUCTBLIP_FLANT5()
    prompt = blip.prepare_prompt(['../Walking_tiger_female.jpg', 'https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg'], 'What is difference between two images?')
    print(blip.get_parsed_output(prompt))
    """
    Output: a tiger and a zebra
    """
    