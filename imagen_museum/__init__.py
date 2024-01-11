import csv
import requests
from io import StringIO
from typing import Union, Optional, Tuple
from PIL import Image

__version__ = "0.0.1_viescore"

DOMAIN = "https://chromaica.github.io/Museum/"

t2i_models= [
      "SD",
      "SDXL",
      "OpenJourney",
      "DeepFloydIF",
      "DALLE2"
    ]

mie_models =  [
      "Glide",
      "SDInpaint",
      "BlendedDiffusion",
      "SDXLInpaint"
    ] 

tie_models = [
      "DiffEdit",
      "MagicBrush",
      "InstructPix2Pix",
      "Prompt2prompt",
      "Text2Live",
      "SDEdit",
      "CycleDiffusion",
      "Pix2PixZero"
    ]

sdig_models = [
      "DreamBooth",
      "DreamBoothLora",
      "TextualInversion",
      "BLIPDiffusion_Gen"
    ]

sdie_models = [
      "PhotoSwap",
      "DreamEdit",
      "BLIPDiffusion_Edit"
    ] 

msdig_models = [
      "DreamBooth",
      "CustomDiffusion",
      "TextualInversion"
    ]

cig_models = [
      "ControlNet",
      "UniControl"
    ]

SC_t2i_examples_1shot = ["https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IG/UniDiffuser/sample_149.jpg"]
SC_tie_examples_1shot = ["https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_139276_1.jpg","https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/Prompt2prompt/sample_139276_1.jpg"]
SC_mie_examples_1shot = SC_tie_examples_1shot
SC_cig_examples_1shot = ["https://chromaica.github.io/Museum/ImagenHub_Control-Guided_IG/input/sample_79_control_hed.jpg", "https://chromaica.github.io/Museum/ImagenHub_Control-Guided_IG/ControlNet/sample_79_control_hed.jpg"]
SC_msdig_examples_1shot = ["https://chromaica.github.io/Museum/ImagenHub_Multi-Concept_IC/input/sample_9.jpg", "https://chromaica.github.io/Museum/ImagenHub_Multi-Concept_IC/CustomDiffusion/sample_9.jpg"]
SC_sdig_examples_1shot = ["https://chromaica.github.io/Museum/ImagenHub_Subject-Driven_IG/input/sample_123.jpg", "https://chromaica.github.io/Museum/ImagenHub_Subject-Driven_IG/SuTI/sample_123.jpg"]
SC_sdie_examples_1shot = ["https://chromaica.github.io/Museum/ImagenHub_Subject-Driven_IE/input/sample_66.jpg", "https://chromaica.github.io/Museum/ImagenHub_Subject-Driven_IE/token/sample_66.jpg", "https://chromaica.github.io/Museum/ImagenHub_Subject-Driven_IE/DreamEdit/sample_66.jpg"]

PQ_examples_1shot = ["https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/CycleDiffusion/sample_187866_1.jpg"]

def fetch_csv_keys(url):
    """
    Fetches a CSV file from a given URL and parses it into a list of keys,
    ignoring the header line.
    """
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses
    
    # Use StringIO to turn the fetched text data into a file-like object
    csv_file = StringIO(response.text)

    # Create a CSV reader
    csv_reader = csv.reader(csv_file)

    # Skip the header
    next(csv_reader, None)

    # Return the list of keys
    return [row[0] for row in csv_reader if row]

def fetch_json_data(url):
    """
    Fetches JSON data from a given URL.
    """
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def fetch_data_and_match(csv_url, json_url):
    """
    Fetches a list of keys from a CSV and then fetches JSON data and matches the keys to the JSON.
    """
    # Fetch keys from CSV
    keys = fetch_csv_keys(csv_url)
    
    # Fetch JSON data
    json_data = fetch_json_data(json_url)
    
    # Extract relevant data using keys
    matched_data = {key: json_data.get(key) for key in keys if key in json_data}

    return matched_data

def fetch_indexes(baselink):
    matched_results = fetch_data_and_match(baselink+"/dataset_lookup.csv", baselink+"/dataset_lookup.json")
    return matched_results

if __name__ == "__main__":
    domain = "https://chromaica.github.io/Museum/"
    baselink = domain + "ImagenHub_Text-Guided_IE"
    matched_results = fetch_indexes()
    for uid, value in matched_results.items():
        print(uid)
        model = "CycleDiffusion"
        image_link = baselink + "/" + model + "/" + uid
        print(image_link)
        instruction = value['instruction']
        print(instruction)
