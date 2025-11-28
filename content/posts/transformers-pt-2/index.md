---
title: "ELI5 Transformers (Part 2) - Generation"
date: 2025-11-11
publishDate: 2025-11-11
draft: true

summary: "(it's glorified sequence classification)"
font_family: "Monospace"
tags: ["AI", "Machine Learning", "ELI5"]
topics: "transformers"

---

{{< katex >}}
![always has been](./featured.jpg "always has been")

*This is part 2 of the ELI5 transformers series, however you do not need to read Part 1 in order to follow the article. Link to the previous article can be found [here](https://yuyiheng.cc/posts/transformers-pt-1/)*

I'm pretty sure you've already heard about someting like this before: 'generative LLM is just slightly advanced auto complete'. Or, something like 'all it does is predicting what's the most likely next word with all the previous words given'.

Today I would like to invite you think of text generation in a very different perspective, at least it's the persceptive I found helped me the most: ***text generation is glorified sequence classification.*** I'm going to walk you through the mechanism of text generation with the source code, and show you how text generation *actually* works.

In this post, you will:
- see the step-by-step process of how texts are generated 

## Recap on transformer architecture
As mentioned in [the previous post](https://yuyiheng.cc/posts/transformers-pt-1/), AI models as we know today computes data in roughly three stages:

{{< mermaid config="theme:mc">}}
flowchart
    n1(["Raw Input"]) --> n2["Embedding"]
    n2 --> n3["Transformer"]
    n3 --> n4["Model output"]
    n4 --> n5(["Final output"])
{{< /mermaid >}}

1. <bullet> Stage 1 converts inputs into vectors so that it's model-readable (**```Embedding```**).
2. <bullet> Stage 2 computes the vector representation of the input. The [previous post](https://yuyiheng.cc/posts/transformers-pt-1/) gave a rough overview of this stage(**```Transformer```**).
3. <bullet> Stage 3 converts outputs from the transformer into human-readable, task-specific format (**```Model output```**).

Stage 1 and 3 are very context-dependent as they are dependent on the type of inputs (text, image, audio etc.,) For image data, this can simply be the RGBA values for each pixel; for text data, this can be a look up table of converting sub-words into matrices. <br>


## Preparation
### Download the model
Open termianl and type-in these codes to install depencies in case you haven't done so:
```sh
pip install transformers
```
Then in terminal, type 'python' to start python interactive REPL:
```sh
python
```
You should see something like this in your terminal:

```sh
Python 3.13.7 | packaged by conda-forge | (main, Sep  3 2025, 14:24:46) [Clang 19.1.7 ] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

Import dependencies:
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
```

*As of November 2025, Pytorch/ Apple still haven't fixed the [memory leak](https://github.com/pytorch/pytorch/issues/91368) issue on Apple Silicon devices (i.e., post-2020 Macbooks). As a result, running models with pytorch for some period of time will gets slower and slower over time. Just for demonstration purpose, I'd recomment manually set pytorch device as 'cpu' because of this. Skip this step if you were confident this won't happen. <br>Since we are just doing demonstrations, we can simly set torch device as 'cpu':<br>*
```python
torch.set_default_device('cpu')  # or 'cuda' if you'd like to use GPU, would not recommend 'mps' (at least for torch<=2.9.1)
```

Download the model:
```python
checkpoint = "Qwen/Qwen3-0.6B"
device = torch.get_default_device()
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, device=device)
```

The openai GPT2 model released on Huggingface doesn't come with some pretty import settings. We'll need to manually amend them first:
```python
pipe.generation_config.pad_token_id = pipe.tokenizer.eos_token_id
pipe.generation_config.bos_token_id = pipe.tokenizer.eos_token_id
pipe.generation_config.decoder_start_token_id = pipe.tokenizer.eos_token_id
```
<br>
You are now all set.

### *Optional: Testing text generation*
We'll first test the text-generation model. Here's a little function to help with text generation. Basically, instead of returning the raw output (list of dictionaries), this function extracts the generated text and prints it directly, just for easier reading.
```python
def generate(text:str, **generate_kwargs) -> None:
    """
    Run text generation pipeline and print out the generated text.
    """
    result = pipe(text, **generate_kwargs)
    print(result[0]['generated_text'])
```

...and now you can use this function to try out text generation yourself:
```python
generate("Who's that Pokemon!?!?", max_new_tokens=5)
```
Since we've set ```do_sample=True``` in our ```pipeline```, text generation is done through random-sampling. Model would generate slightly different answer everytime we run our ```generation``` function.<br>
Let's run 5 of them:
```python
for _ in range(5):
    generate("Who's that Pokemon!?!?", max_new_tokens=5)
```
Your output would be very different from mine:
```console
Who's that Pokemon? Haha… This one
Who's that Pokemon? I can't really recall
Who's that Pokemon? It'll have a head
Who's that Pokemon? I'm going to catch
Who's that Pokemon?
The first week has
```
*You can generate much longer text by setting 'max_new_tokens' to a higher number, note that the model basically generates typical Aİ slops when it's too large:*
```python
generate("Who's that Pokemon!?!?", max_new_tokens=1024)
```



### 1 - Inputs <-> outputs

### 2 - 'Task Head'

## Model generation vs text classification
(Here is where I write about LM head)

## Popular text generation strategies

## 预告？？？parallel training