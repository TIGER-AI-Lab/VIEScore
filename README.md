# VIEScore
[![arXiv](https://img.shields.io/badge/arXiv-2312.14867-b31b1b.svg)](https://arxiv.org/abs/2312.14867)

[![contributors](https://img.shields.io/github/contributors/TIGER-AI-Lab/VIEScore)](https://github.com/TIGER-AI-Lab/VIEScore/graphs/contributors)
[![license](https://img.shields.io/github/license/TIGER-AI-Lab/VIEScore.svg)](https://github.com/TIGER-AI-Lab/VIEScore/blob/main/LICENSE)
[![GitHub](https://img.shields.io/github/stars/TIGER-AI-Lab/VIEScore?style=social)](https://github.com/TIGER-AI-Lab/VIEScore)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FTIGER-AI-Lab%2FVIEScore&count_bg=%23C83DB9&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false)](https://hits.seeyoufarm.com)

This repository hosts our work's code [VIEScore: Towards Explainable Metrics for Conditional Image Synthesis Evaluation](https://tiger-ai-lab.github.io/VIEScore/).

VIEScore is a Visual Instruction-guided Explainable metric for evaluating any conditional image generation tasks.

<div align="center">
 🔥 🔥 🔥 Check out our <a href = "https://tiger-ai-lab.github.io/VIEScore/">[Project Page]</a> for more results and analysis!
</div>

<div align="center">
<img src="https://github.com/TIGER-AI-Lab/VIEScore/blob/gh-pages/static/images/teaser.png" width="100%">
  Metrics in the future would provide the score and the rationale, enabling the understanding of each judgment. Which method (VIEScore or traditional metrics) is “closer” to the human perspective?
</div>

## 📰 News
* 2024 May 15: VIEScore is accepted to ACL2024! 
* 2024 Jan 11: Code is released!
* 2023 Dec 24: Paper available on [Arxiv](https://arxiv.org/abs/2312.14867). Code coming Soon!


## Project Structure

* `imagen_museum` : helpers to fetch image data from [ImagenMuseum](https://chromaica.github.io/)
* `mllm_tools`: Plug-and-Play MLLMs.
* `_questions`: prompt folder
* `_answers`: results folder
* `run.py`: script to run VIEScore. 
* `clean_result.py`: script to clear nonsense results according to `banned_reasonings.txt`. 
* `count_entries.py`: script to count the number of entries.

## Running VIEScore

![](https://tiger-ai-lab.github.io/VIEScore/static/images/method.png)

```shell
$ python3 run.py --help
usage: run.py [-h] [--task {tie,mie,t2i,cig,sdig,msdig,sdie}] [--mllm {gpt4v,llava,blip2,fuyu,qwenvl,cogvlm,instructblip,openflamingo}] [--setting {0shot,1shot}] [--context_file CONTEXT_FILE]
              [--guess_if_cannot_parse]

Run different task on VIEScore.

optional arguments:
  -h, --help            show this help message and exit
  --task {tie,mie,t2i,cig,sdig,msdig,sdie}
                        Select the task to run
  --mllm {gpt4v,llava,blip2,fuyu,qwenvl,cogvlm,instructblip,openflamingo}
                        Select the MLLM model to use
  --setting {0shot,1shot}
                        Select the incontext learning setting
  --context_file CONTEXT_FILE
                        Which context file to use.
  --guess_if_cannot_parse
                        Guess a value if the output cannot be parsed.
```

For example, you can run:
```shell
python3 run.py --task t2i --mllm gpt4v --setting 0shot --context_file context.txt
```

* Available context files are in `_questions` folder.

After running the experiment, you can count the results or clean it up:

```shell
python3 count_entries.py <your_answers_dir>
```

```shell
python3 clean_result.py <your_answers_dir>
```

## Citation

Please kindly cite our paper if you use our code, data, models or results:

```bibtex
@misc{ku2023viescore,
                title={VIEScore: Towards Explainable Metrics for Conditional Image Synthesis Evaluation}, 
                author={Max Ku and Dongfu Jiang and Cong Wei and Xiang Yue and Wenhu Chen},
                year={2023},
                eprint={2312.14867},
                archivePrefix={arXiv},
                primaryClass={cs.CV}
            }
```
