# VIEScore (Paper Implementation)

Paper implementation of VIEScore. (Tied to Imagen Museum).

## Project Structure

* `imagen_museum` : helpers to fetch image data from [ImagenMuseum](https://chromaica.github.io/)
* `mllm_tools`: Plug-and-Play MLLMs codebase.
* `_questions`: prompt folder
* `_answers`: results folder
* `run.py`: script to run VIEScore. 
* `clean_result.py`: script to clear nonsense results according to `banned_reasonings.txt`. 
* `count_entries.py`: script to count the number of entries.

## Running VIEScore

![](https://tiger-ai-lab.github.io/VIEScore/static/images/method.png)

```shell
$ python3 run.py --help
usage: run.py [-h] [--task {tie,mie,t2i,cig,sdig,msdig,sdie}] [--mllm {gpt4v, gpt4o, llava,blip2,fuyu,qwenvl,cogvlm,instructblip,openflamingo, gemini}] [--setting {0shot,1shot}] [--context_file CONTEXT_FILE]
              [--guess_if_cannot_parse]

Run different task on VIEScore.

optional arguments:
  -h, --help            show this help message and exit
  --task {tie,mie,t2i,cig,sdig,msdig,sdie}
                        Select the task to run
  --mllm {gpt4v, gpt4o, llava,blip2,fuyu,qwenvl,cogvlm,instructblip,openflamingo, gemini}
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

## Compute Correlations

Refer to `analyze_json.ipynb` notebook.

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
