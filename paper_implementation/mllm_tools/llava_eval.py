import os
import tempfile
import requests
from PIL import Image
from io import BytesIO
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from .run_llava import eval_model
from mllm_tools.utils import merge_images
from typing import List

class Llava():
    merged_image_files = []
    def __init__(self, model_path:str="liuhaotian/llava-v1.5-7b") -> None:
        """Llava model wrapper
        requires: build from source https://github.com/haotian-liu/LLaVA.git
        Args:
            model_path (str): Llava model name, e.g. "liuhaotian/llava-v1.5-7b"
        """
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path)
        )
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len
    
    def prepare_prompt(self, image_links: List = [], text_prompt: str = ""):
        if type(image_links) == str:
            image_links = [image_links]
        if len(image_links) > 1:
            # Merge images if there are more than one, because llava only accepts one image
            tmp_image_file = merge_images(image_links)
            image_file = tmp_image_file
            import os
        else:
            tmp_image_file = None
            image_file = None if len(image_links) == 0 else image_links[0]
        args = type('Args', (), {
            "model_path": self.model_path,
            "model_base": None,
            "model_name": get_model_name_from_path(self.model_path),
            "query": text_prompt,
            "conv_mode": None,
            "image_file": image_file,
            "sep": ",",
            "temperature": 0.2,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512,
        })()
        return args

    def get_parsed_output(self, args):
        output = eval_model(args, self.tokenizer, self.model, self.image_processor, self.context_len)
        return output

    def __del__(self):
        for image_file in self.merged_image_files:
            if os.path.exists(image_file):
                os.remove(image_file)
    
if __name__ == "__main__":
    model_path = "liuhaotian/llava-v1.5-7b"
    llava = Llava(model_path)
    prompt = llava.prepare_prompt(['https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Walking_tiger_female.jpg/1920px-Walking_tiger_female.jpg', 'https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg'], 'What is difference between two images?')
    print(llava.get_parsed_output(prompt))