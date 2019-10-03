# Gradient Descent: The Ultimate Optimizer

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

| ⚠️ WARNING: THIS IS NOT MY WORK ⚠️ |
| --- |

This repository contains the paper and code to the paper [Gradient Descent:
The Ultimate Optimizer](https://arxiv.org/abs/1909.13371).

I couldn't find the code (which is found in the appendix at the end of the
paper) anywhere on the web. What I present here is the code of the paper with
instructions on how to set it up.

Getting the code in a runnable state required some fixes on my part so the
code might be slightly different than that presented in the paper.

## Set up 

```sh
git clone https://github.com/Rainymood/Gradient-Descent-The-Ultimate-Optimizer 
cd Gradient-Descent-The-Ultimate-Optimizer
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

When you are done you can exit the virtualenv with 

```shell
deactivate
```
