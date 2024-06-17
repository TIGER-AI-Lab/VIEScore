import os
import logging
from imagen_museum.utils import (
    read_files_to_string,
    get_file_path,
    write_entry_to_json_file,
    check_key_in_json,
)
from imagen_museum import (
    fetch_indexes,
    t2i_models,
    tie_models,
    mie_models,
    sdig_models,
    sdie_models,
    msdig_models,
    cig_models,
)
from imagen_museum import (
    SC_tie_examples_1shot,
    SC_cig_examples_1shot,
    SC_sdig_examples_1shot,
    SC_msdig_examples_1shot,
    SC_mie_examples_1shot,
    SC_sdie_examples_1shot,
    SC_t2i_examples_1shot,
    PQ_examples_1shot,
)

from imagen_museum import DOMAIN as imagen_museum_domain

from typing import Callable, List
import argparse
import time
import random

# Some init settings, override them with command line arguments later.
#===============================================================================
context_file = None
mllm_model = None  # initalize it later
search_from_dir = (
    "prompts_<SETTING>"
)
save_to_dir = "results_<MODEL>_<SETTING>"
setting = "0shot"
SC_file_name = "SC.json"
PQ_file_name = "PQ.json"
max_tries = 3
guess_if_cannot_parse = False

#===============================================================================

def iterate_infer(prompt, uid, text_prompt, image_prompt, target_file):
    is_verified = False
    tries = 0
    while not is_verified and tries < max_tries:
        result = mllm_model.get_parsed_output(prompt)
        is_verified = write_entry_to_json_file(result, uid, text_prompt, image_prompt, target_file, give_up_parsing=guess_if_cannot_parse)
        tries += 1
        if is_verified == True:
            logging.info(f"saved {uid}")
        else:
            print(image_prompt, result)


def infer_tie_common(
    modelname, rule_file_paths, target_file_name, prefix_images_links=None
):
    task_name = "ImagenHub_Text-Guided_IE"
    baselink = imagen_museum_domain + task_name
    init_text_prompt = read_files_to_string(rule_file_paths)

    matched_results = fetch_indexes(baselink)

    for uid, value in matched_results.items():
        target_file = os.path.join(save_to_dir, task_name, modelname, target_file_name)
        if check_key_in_json(target_file, key=uid):
            logging.info(f"skipping {uid}")
            continue
        input_image_link = baselink + "/" + "input" + "/" + uid
        model_image_link = baselink + "/" + modelname + "/" + uid
        if target_file_name == SC_file_name:
            image_prompt = [input_image_link, model_image_link]
            text_prompt = init_text_prompt.replace(
                "<instruction>", value["instruction"]
            )
        elif target_file_name == PQ_file_name:
            image_prompt = [model_image_link]
            text_prompt = init_text_prompt
        if prefix_images_links is not None:
            image_prompt = prefix_images_links + image_prompt  # concatenate the lists
        prompt = mllm_model.prepare_prompt(image_prompt, text_prompt)

        iterate_infer(prompt, uid, text_prompt, image_prompt, target_file)


def infer_tie_SC(modelname):
    file_paths = [
        context_file,
        get_file_path("two_image_edit_rule.txt", search_from_dir),
        get_file_path("tie_rule_SC.txt", search_from_dir),
    ]
    prefix = SC_tie_examples_1shot if setting == "1shot" else None
    infer_tie_common(modelname, file_paths, SC_file_name, prefix_images_links=prefix)


def infer_tie_PQ(modelname):
    file_paths = [context_file, get_file_path("rule_PQ.txt", search_from_dir)]
    prefix = PQ_examples_1shot if setting == "1shot" else None
    infer_tie_common(modelname, file_paths, PQ_file_name, prefix_images_links=prefix)


def infer_mie_common(
    modelname, rule_file_paths, target_file_name, prefix_images_links=None
):
    task_name = "ImagenHub_Mask-Guided_IE"
    baselink = imagen_museum_domain + task_name
    init_text_prompt = read_files_to_string(rule_file_paths)

    matched_results = fetch_indexes(baselink)

    for uid, value in matched_results.items():
        target_file = os.path.join(save_to_dir, task_name, modelname, target_file_name)
        if check_key_in_json(target_file, key=uid):
            logging.info(f"skipping {uid}")
            continue
        input_image_link = baselink + "/" + "input" + "/" + uid
        model_image_link = baselink + "/" + modelname + "/" + uid
        if target_file_name == SC_file_name:
            image_prompt = [input_image_link, model_image_link]
            text_prompt = init_text_prompt.replace(
                "<instruction>", value["instruction"]
            )
        elif target_file_name == PQ_file_name:
            image_prompt = [model_image_link]
            text_prompt = init_text_prompt
        if prefix_images_links is not None:
            image_prompt = prefix_images_links + image_prompt  # concatenate the lists
        prompt = mllm_model.prepare_prompt(image_prompt, text_prompt)

        iterate_infer(prompt, uid, text_prompt, image_prompt, target_file)


def infer_mie_SC(modelname):
    file_paths = [
        context_file,
        get_file_path("two_image_edit_rule.txt", search_from_dir),
        get_file_path("mie_rule_SC.txt", search_from_dir),
    ]
    prefix = SC_mie_examples_1shot if setting == "1shot" else None
    infer_mie_common(modelname, file_paths, SC_file_name, prefix_images_links=prefix)


def infer_mie_PQ(modelname):
    file_paths = [context_file, get_file_path("rule_PQ.txt", search_from_dir)]
    prefix = PQ_examples_1shot if setting == "1shot" else None
    infer_mie_common(modelname, file_paths, PQ_file_name, prefix_images_links=prefix)


def infer_t2i_common(
    modelname, rule_file_paths, target_file_name, prefix_images_links=None
):
    task_name = "ImagenHub_Text-Guided_IG"
    baselink = imagen_museum_domain + task_name
    init_text_prompt = read_files_to_string(rule_file_paths)

    matched_results = fetch_indexes(baselink)

    for uid, value in matched_results.items():
        target_file = os.path.join(save_to_dir, task_name, modelname, target_file_name)
        if check_key_in_json(target_file, key=uid):
            logging.info(f"skipping {uid}")
            continue
        model_image_link = baselink + "/" + modelname + "/" + uid
        image_prompt = [model_image_link]
        if target_file_name == SC_file_name:
            text_prompt = init_text_prompt.replace("<prompt>", value["prompt"])
        elif target_file_name == PQ_file_name:
            text_prompt = init_text_prompt
        if prefix_images_links is not None:
            image_prompt = prefix_images_links + image_prompt  # concatenate the lists
        prompt = mllm_model.prepare_prompt(image_prompt, text_prompt)

        iterate_infer(prompt, uid, text_prompt, image_prompt, target_file)


def infer_t2i_SC(modelname):
    file_paths = [
        context_file,
        get_file_path("one_image_gen_rule.txt", search_from_dir),
        get_file_path("t2i_rule_SC.txt", search_from_dir),
    ]
    prefix = SC_t2i_examples_1shot if setting == "1shot" else None
    infer_t2i_common(modelname, file_paths, SC_file_name, prefix_images_links=prefix)


def infer_t2i_PQ(modelname):
    file_paths = [context_file, get_file_path("rule_PQ.txt", search_from_dir)]
    prefix = PQ_examples_1shot if setting == "1shot" else None
    infer_t2i_common(modelname, file_paths, PQ_file_name, prefix_images_links=prefix)


def infer_cig_common(
    modelname, rule_file_paths, target_file_name, prefix_images_links=None
):
    task_name = "ImagenHub_Control-Guided_IG"
    baselink = imagen_museum_domain + task_name
    init_text_prompt = read_files_to_string(rule_file_paths)

    matched_results = fetch_indexes(baselink)

    for uid, value in matched_results.items():
        target_file = os.path.join(save_to_dir, task_name, modelname, target_file_name)
        if check_key_in_json(target_file, key=uid):
            logging.info(f"skipping {uid}")
            continue
        input_image_link = baselink + "/" + "input" + "/" + uid
        model_image_link = baselink + "/" + modelname + "/" + uid

        if target_file_name == SC_file_name:
            image_prompt = [input_image_link, model_image_link]
            text_prompt = init_text_prompt.replace("<prompt>", value["text"])
        elif target_file_name == PQ_file_name:
            image_prompt = [model_image_link]
            text_prompt = init_text_prompt
        if prefix_images_links is not None:
            image_prompt = prefix_images_links + image_prompt  # concatenate the lists
        prompt = mllm_model.prepare_prompt(image_prompt, text_prompt)

        iterate_infer(prompt, uid, text_prompt, image_prompt, target_file)


def infer_cig_SC(modelname):
    file_paths = [
        context_file,
        get_file_path("control_image_gen_rule.txt", search_from_dir),
        get_file_path("cig_rule_SC.txt", search_from_dir),
    ]
    prefix = SC_cig_examples_1shot if setting == "1shot" else None
    infer_cig_common(modelname, file_paths, SC_file_name, prefix_images_links=prefix)


def infer_cig_PQ(modelname):
    file_paths = [context_file, get_file_path("rule_PQ.txt", search_from_dir)]
    prefix = PQ_examples_1shot if setting == "1shot" else None
    infer_cig_common(modelname, file_paths, PQ_file_name, prefix_images_links=prefix)


def infer_sdig_common(
    modelname, rule_file_paths, target_file_name, prefix_images_links=None
):
    task_name = "ImagenHub_Subject-Driven_IG"
    baselink = imagen_museum_domain + task_name
    init_text_prompt = read_files_to_string(rule_file_paths)

    matched_results = fetch_indexes(baselink)

    for uid, value in matched_results.items():
        target_file = os.path.join(save_to_dir, task_name, modelname, target_file_name)
        if check_key_in_json(target_file, key=uid):
            logging.info(f"skipping {uid}")
            continue
        input_image_link = baselink + "/" + "input" + "/" + uid
        model_image_link = baselink + "/" + modelname + "/" + uid
        if target_file_name == SC_file_name:
            image_prompt = [input_image_link, model_image_link]
            text_prompt = init_text_prompt.replace(
                "<prompt>", value["prompt"].replace("<token> ", "")
            )
        elif target_file_name == PQ_file_name:
            image_prompt = [model_image_link]
            text_prompt = init_text_prompt
        if prefix_images_links is not None:
            image_prompt = prefix_images_links + image_prompt  # concatenate the lists
        prompt = mllm_model.prepare_prompt(image_prompt, text_prompt)

        iterate_infer(prompt, uid, text_prompt, image_prompt, target_file)


def infer_sdig_SC(modelname):
    file_paths = [
        context_file,
        get_file_path("subject_image_gen_rule.txt", search_from_dir),
        get_file_path("sdig_rule_SC.txt", search_from_dir),
    ]
    prefix = SC_sdig_examples_1shot if setting == "1shot" else None
    infer_sdig_common(modelname, file_paths, SC_file_name, prefix_images_links=prefix)


def infer_sdig_PQ(modelname):
    file_paths = [context_file, get_file_path("rule_PQ.txt", search_from_dir)]
    prefix = PQ_examples_1shot if setting == "1shot" else None
    infer_sdig_common(modelname, file_paths, PQ_file_name, prefix_images_links=prefix)


def infer_sdie_common(
    modelname, rule_file_paths, target_file_name, prefix_images_links=None
):
    task_name = "ImagenHub_Subject-Driven_IE"
    baselink = imagen_museum_domain + task_name
    init_text_prompt = read_files_to_string(rule_file_paths)

    matched_results = fetch_indexes(baselink)

    for uid, value in matched_results.items():
        target_file = os.path.join(save_to_dir, task_name, modelname, target_file_name)
        if check_key_in_json(target_file, key=uid):
            logging.info(f"skipping {uid}")
            continue
        input_image_link = baselink + "/" + "input" + "/" + uid
        token_image_link = baselink + "/" + "token" + "/" + uid
        model_image_link = baselink + "/" + modelname + "/" + uid
        if target_file_name == SC_file_name:
            image_prompt = [input_image_link, token_image_link, model_image_link]
            text_prompt = init_text_prompt.replace(
                "<subject>", value["subject"].replace("_", " ")
            )
        elif target_file_name == PQ_file_name:
            image_prompt = [model_image_link]
            text_prompt = init_text_prompt
        if prefix_images_links is not None:
            image_prompt = prefix_images_links + image_prompt  # concatenate the lists
        prompt = mllm_model.prepare_prompt(image_prompt, text_prompt)

        iterate_infer(prompt, uid, text_prompt, image_prompt, target_file)


def infer_sdie_SC(modelname):
    file_paths = [
        context_file,
        get_file_path("subject_image_edit_rule.txt", search_from_dir),
        get_file_path("sdie_rule_SC.txt", search_from_dir),
    ]
    prefix = SC_sdie_examples_1shot if setting == "1shot" else None
    infer_sdie_common(modelname, file_paths, SC_file_name, prefix_images_links=prefix)


def infer_sdie_PQ(modelname):
    file_paths = [context_file, get_file_path("rule_PQ.txt", search_from_dir)]
    prefix = PQ_examples_1shot if setting == "1shot" else None
    infer_sdie_common(modelname, file_paths, PQ_file_name, prefix_images_links=prefix)


def infer_msdig_common(
    modelname, rule_file_paths, target_file_name, prefix_images_links=None
):
    task_name = "ImagenHub_Multi-Concept_IC"
    baselink = imagen_museum_domain + task_name
    init_text_prompt = read_files_to_string(rule_file_paths)

    matched_results = fetch_indexes(baselink)

    for uid, value in matched_results.items():
        target_file = os.path.join(save_to_dir, task_name, modelname, target_file_name)
        if check_key_in_json(target_file, key=uid):
            logging.info(f"skipping {uid}")
            continue
        input_image_link = baselink + "/" + "input" + "/" + uid
        model_image_link = baselink + "/" + modelname + "/" + uid
        if target_file_name == SC_file_name:
            image_prompt = [input_image_link, model_image_link]
            text_prompt = init_text_prompt.replace("<prompt>", value["prompt"])
        elif target_file_name == PQ_file_name:
            image_prompt = [model_image_link]
            text_prompt = init_text_prompt
        if prefix_images_links is not None:
            image_prompt = prefix_images_links + image_prompt  # concatenate the lists
        prompt = mllm_model.prepare_prompt(image_prompt, text_prompt)

        iterate_infer(prompt, uid, text_prompt, image_prompt, target_file)


def infer_msdig_SC(modelname):
    file_paths = [
        context_file,
        get_file_path("multi_subject_image_gen_rule.txt", search_from_dir),
        get_file_path("msdig_rule_SC.txt", search_from_dir),
    ]
    prefix = SC_msdig_examples_1shot if setting == "1shot" else None
    infer_msdig_common(modelname, file_paths, SC_file_name, prefix_images_links=prefix)


def infer_msdig_PQ(modelname):
    file_paths = [context_file, get_file_path("rule_PQ.txt", search_from_dir)]
    prefix = PQ_examples_1shot if setting == "1shot" else None
    infer_msdig_common(modelname, file_paths, PQ_file_name, prefix_images_links=prefix)


def run_VIEScore(modelnames: List, SC_fn: Callable, PQ_fn: Callable):
    logging.info("Do not use threading")
    random.shuffle(modelnames)
    for modelname in modelnames:
        SC_fn(modelname)
        PQ_fn(modelname)
        logging.info("run_tie for model %s: all done", modelname)
    logging.info("Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run different task on VIEScore.")
    parser.add_argument(
        "--task",
        type=str,
        choices=["tie", "mie", "t2i", "cig", "sdig", "msdig", "sdie"],
        help="Select the task to run",
    )
    parser.add_argument(
        "--mllm",
        type=str,
        choices=["gpt4v", "gpt4o", "llava", "blip2", "fuyu", "qwenvl", "cogvlm", "instructblip", "openflamingo", "gemini"],
        required=False,
        default="gpt4v",
        help="Select the MLLM model to use",
    )
    parser.add_argument(
        "--setting",
        type=str,
        choices=["0shot", "1shot"],
        required=False,
        default="0shot",
        help="Select the incontext learning setting",
    )
    parser.add_argument(
        "--context_file",
        type=str,
        required=False,
        default="./_questions/context.txt",
        help="Which context file to use."
    )
    parser.add_argument(
        "--guess_if_cannot_parse",
        action='store_true',
        help="Guess a value if the output cannot be parsed."
    )

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    args = parser.parse_args()
    if args.guess_if_cannot_parse:
        guess_if_cannot_parse = True
    setting = args.setting
    save_to_dir = save_to_dir.replace("<MODEL>", args.mllm)
    save_to_dir = save_to_dir.replace("<SETTING>", setting)
    search_from_dir = search_from_dir.replace("<SETTING>", setting)

    if args.context_file != "context.txt":
        context_file = args.context_file
    context_file = get_file_path(os.path.join("_questions", context_file))
    save_to_dir = os.path.join("_answers", save_to_dir)

    print("="*80)
    print("Using context file:", context_file)
    print("MLLM Model used:", args.mllm)
    print("Incontext learning setting:", setting)
    print("Save to dir:", save_to_dir)
    print("Guess if cannot parse:", guess_if_cannot_parse)
    print("="*80)

    # Override the model.
    if args.mllm == "gpt4v":
        from mllm_tools.openai import GPT4v
        try:
            gpt4v_keys = get_file_path("YOUR_API_KEY.env")
            # Alternatively you can do something like this:
            #gpt4v_keys = [get_file_path("key1.env"), get_file_path("key2.env"), get_file_path("key3.env")]
        except:
            print("No secret.env file found. Please create one with your OpenAI API key.")
        mllm_model = GPT4v(gpt4v_keys, are_images_encoded=True)
    elif args.mllm == 'gpt4o':
        from mllm_tools.openai import GPT4o
        try:
            gpt4v_keys = get_file_path("YOUR_API_KEY.env")
            # Alternatively you can do something like this:
            #gpt4v_keys = [get_file_path("key1.env"), get_file_path("key2.env"), get_file_path("key3.env")]
        except:
            print("No secret.env file found. Please create one with your OpenAI API key.")
        mllm_model = GPT4o(gpt4v_keys, are_images_encoded=True)
    elif args.mllm == 'gemini':
        from mllm_tools.gemini import Gemini
        mllm_model = Gemini()
    if args.mllm == 'llava':
        from mllm_tools.llava_eval import Llava
        mllm_model = Llava()
    elif args.mllm == 'blip2':
        from mllm_tools.blip_flant5_eval import BLIP_FLANT5
        mllm_model = BLIP_FLANT5()
    elif args.mllm == 'fuyu':
        from mllm_tools.fuyu_eval import Fuyu
        mllm_model = Fuyu()
    elif args.mllm == 'qwenvl':
        from mllm_tools.qwenVL_eval import QwenVL
        mllm_model = QwenVL()
    elif args.mllm == 'cogvlm':
        from mllm_tools.cogvlm_eval import CogVLM
        mllm_model = CogVLM()
    elif args.mllm == 'instructblip':
        from mllm_tools.instructblip_eval import INSTRUCTBLIP_FLANT5
        mllm_model = INSTRUCTBLIP_FLANT5()
    elif args.mllm == 'openflamingo':
        from mllm_tools.openflamingo_eval import OpenFlamingo
        mllm_model = OpenFlamingo()

    # Run the appropriate task based on the argument
    if args.task == "tie":
        run_VIEScore(tie_models, infer_tie_SC, infer_tie_PQ)
    elif args.task == "mie":
        run_VIEScore(mie_models, infer_mie_SC, infer_mie_PQ)
    elif args.task == "t2i":
        run_VIEScore(t2i_models, infer_t2i_SC, infer_t2i_PQ)
    elif args.task == "cig":
        run_VIEScore(cig_models, infer_cig_SC, infer_cig_PQ)
    elif args.task == "sdig":
        run_VIEScore(sdig_models, infer_sdig_SC, infer_sdig_PQ)
    elif args.task == "sdie":
        run_VIEScore(sdie_models, infer_sdie_SC, infer_sdie_PQ)
    elif args.task == "msdig":
        run_VIEScore(msdig_models, infer_msdig_SC, infer_msdig_PQ)
    else:
        print("No valid task selected. Use --help for more information.")
