import requests
import json
from typing import List

def get_api_key(file_path):
    # Read the API key from the first line of the file
    with open(file_path, 'r') as file:
        return file.readline().strip()

class GPT4v():
    def __init__(self, api_key_path='keys/secret_thirdparty.env'):
        self.url = "https://plus.bothyouandme.com/v1/chat/completions"
        api_key = get_api_key(api_key_path)
        if not api_key:
            print("API key not found.")
            exit(1)

        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def prepare_prompt(self, image_links: List = [], text_prompt: str = ""):
        prompt = ""
        if isinstance(image_links, str):
            image_links = [image_links]
        for image_link in image_links:
            prompt += f"[GPT-4 Vision]({image_link}) "
        prompt += text_prompt
        return prompt

    def get_parsed_output(self, prompt):
        data = {
            "model": "gpt-4-vision-preview",
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": f"{prompt}"
                }
            ]
        }
        response = requests.post(self.url, json=data, headers=self.headers, stream=True)
        return self.extract_response(response)

    def extract_response(self, response):
        final_response = ""
        for line in response.iter_lines():
            if line == b'data: [DONE]':
                break
            elif line.startswith(b'data: '):
                # Remove the 'data: ' prefix and parse the JSON
                json_str = line.decode('utf-8')[6:]
                try:
                    parsed_line = json.loads(json_str)
                    content_chunk = parsed_line.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    final_response += content_chunk
                except json.JSONDecodeError:
                    print("Failed to decode line as JSON:", json_str)
                    continue
        return final_response

if __name__ == "__main__":
    model = GPT4v()
    prompt = model.prepare_prompt(['https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_1.jpg', 'https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg'], 'What is difference between two images?')
    print("prompt : \n", prompt)
    res = model.get_parsed_output(prompt)
    print("result : \n", res)