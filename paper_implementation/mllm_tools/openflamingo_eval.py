import torch
from open_flamingo import create_model_and_transforms
from typing import List
from mllm_tools.utils import merge_images, load_images
from huggingface_hub import hf_hub_download
import torch

class OpenFlamingo():
    xatten_map = {
        "anas-awadalla/mpt-1b-redpajama-200b": 1,
        "anas-awadalla/mpt-1b-redpajama-200b-dolly": 1,
        "togethercomputer/RedPajama-INCITE-Base-3B-v1": 2,
        "togethercomputer/RedPajama-INCITE-Instruct-3B-v1": 2,
        "anas-awadalla/mpt-7b": 4,
    }
    hf_op_checkpoint_map = {
        "anas-awadalla/mpt-1b-redpajama-200b": "openflamingo/OpenFlamingo-3B-vitl-mpt1b",
        "anas-awadalla/mpt-1b-redpajama-200b-dolly": "openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct",
        "togethercomputer/RedPajama-INCITE-Base-3B-v1": "openflamingo/OpenFlamingo-4B-vitl-rpj3b",
        "togethercomputer/RedPajama-INCITE-Instruct-3B-v1": "openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct",
        "anas-awadalla/mpt-7b": "openflamingo/OpenFlamingo-9B-vitl-mpt7b",
    }
    def __init__(self, model_id="togethercomputer/RedPajama-INCITE-Instruct-3B-v1"):
        """
        requires: pip install open-flamingo
        Args:
            model_id (str): OpenFlamingo model name, e.g. "togethercomputer/RedPajama-INCITE-Instruct-3B-v1"
        """
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path=model_id,
            tokenizer_path=model_id,
            cross_attn_every_n_layers=self.xatten_map[model_id],
        )
        self.model = model
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.tokenizer.paddding_side = "left"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        checkpoint_path = hf_hub_download(self.hf_op_checkpoint_map[model_id], "checkpoint.pt")
        model.load_state_dict(torch.load(checkpoint_path), strict=False)
        self.model.eval()
        
    def prepare_prompt(self, image_links:List[str], text_prompt: List[str]):
        inputs = self._prepare_prompt([image_links], text_prompt)
        return inputs
    
    def _prepare_prompt(self, image_links:List[List[str]], text_prompt: List[str]):
        if type(image_links) == str:
            image_links = [[image_links]]
        if isinstance(text_prompt, str):
            text_prompt = [text_prompt]
            
        assert len(image_links) == len(text_prompt), "Number of images and text prompts should be the same"
        merged_images = [merge_images(_image_links) for _image_links in image_links]
        for i, im in enumerate(merged_images):
            im.save("./merged_image_{}.jpg".format(i))
        
        prompt = ""
        for i in range(len(text_prompt)):
            prefix = "<image>\n" if i < len(merged_images) else ""
            if i == len(text_prompt) - 1:
                prompt += prefix + text_prompt[i]
            else:
                prompt += prefix + text_prompt[i] + "\n<|endofchunk|>\n"
        prompt = prompt.strip(' \n')
        
        vision_x = [self.image_processor(image).unsqueeze(0) for image in merged_images]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)
        lang_x = self.tokenizer(prompt, return_tensors="pt")
        return {"vision_x": vision_x, "lang_x": lang_x}


    def get_parsed_output(self, inputs):
        vision_x = inputs["vision_x"].to(self.device)
        lang_x = {k: v.to(self.device) for k, v in inputs["lang_x"].items()}
        generated_text = self.model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"],
            attention_mask=lang_x["attention_mask"],
            max_new_tokens=128,
            num_beams=3,
        )
        decoded_text = self.tokenizer.decode(generated_text[0][lang_x["input_ids"].shape[1]:], skip_special_tokens=True)
        return decoded_text.strip(' \n')
    
    
if __name__ == "__main__":
    model = OpenFlamingo()
    # 0 shot
    # prompt = model.prepare_prompt([['../Walking_tiger_female.jpg']], ['What is in the image?'], num_shots=0)
    # 1 shot
    prompt = model.prepare_prompt([['../Walking_tiger_female.jpg'], ["https://hips.hearstapps.com/hmg-prod/images/rabbit-breeds-american-white-1553635287.jpg?crop=0.976xw:0.651xh;0.0242xw,0.291xh&resize=980:*"]], ['What is in the image? A tiger.', 'What is in the image?'])
    # 2 shot
    # prompt = model.prepare_prompt([['../Walking_tiger_female.jpg'], ['https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg'], ["https://hips.hearstapps.com/hmg-prod/images/rabbit-breeds-american-white-1553635287.jpg?crop=0.976xw:0.651xh;0.0242xw,0.291xh&resize=980:*"]], ['What is in the image? A tiger.', 'What is in the image? A zabra', 'What is in the image?'])
    print(model.get_parsed_output(prompt))
    """
    Output: a tiger and a zebra
    """
    