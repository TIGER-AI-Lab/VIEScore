import torch
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
from io import BytesIO
from typing import List
from mllm_tools.utils import merge_images

class CogVLM():
    def __init__(self, model_id:str="THUDM/cogvlm-chat-hf") -> None:
        """
        requires: pip install xformers
        Args:
            model_id (str): CogVLM model name, e.g. "THUDM/cogvlm-chat-hf"
        """
        self.model_id = model_id
        self.tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to('cuda').eval()
    
    
    def prepare_prompt(self, image_links: List = [], text_prompt: str = ""):
        if type(image_links) == str:
            image_links = [image_links]
        merged_image = merge_images(image_links)
        inputs = self.model.build_conversation_input_ids(self.tokenizer, query=text_prompt, history=[], images=[merged_image])  # chat mode
        return inputs
    
    def get_parsed_output(self, inputs):
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
        }
        gen_kwargs = {"max_length": 2048, "do_sample": False, 'no_repeat_ngram_size': 3, 'early_stopping': True}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output

if __name__ == "__main__":
    model = CogVLM()
    prompt = model.prepare_prompt(['../Walking_tiger_female.jpg', 'https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg'], 'What is difference between two images?')
    print(model.get_parsed_output(prompt))
    """
    Output: In the image on the left, a tiger is walking through a grassy field. In the image on the right, a zebra is walking through a grassy field.
    """
    