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

Today I would like to invite you think of text generation in a very different perspective, at least it's the persceptive I found helped me the most: ***text generation is glorified sequence classification.*** I'm going to walk your way through the text generation process, step-by-step, one word at a time, showing you how text generation *actually* works.



<p style="font-size:250%">FINISH THIS LATER!!</p>

## Recap on transformer architecture
As mentioned in [the previous post](https://yuyiheng.cc/posts/transformers-pt-1/), AI models as we know today computes data in roughly three stages:

{{< mermaid config="theme:mc">}}
flowchart LR
    n1(["Raw Input"]) --> n2["Embedding"]
    n2 --> n3["Transformer"]
    n3 --> n4["Model output"]
    n4 --> n5(["Final output"])
    classDef default fill: transparent, bg-color: transparent
{{< /mermaid >}}

1. <bullet> Stage 1 converts inputs into vectors so that it's model-readable (**```Embedding```**).
2. <bullet> Stage 2 computes the vector representation of the input. This is the main focus of the [previous post](https://yuyiheng.cc/posts/transformers-pt-1/) (**```Transformer```**).
3. <bullet> Stage 3 converts outputs from the transformer into human-readable, task-specific format (**```Model output```**).

Stage 1 and 3 are very context-dependent as they are dependent on the type of inputs (text, image, audio etc.,) For image data, this can simply be the RGBA values for each pixel; for text data, this can be a look up table of converting sub-words into matrices. <br>

The model that's going to be used in this demo is [Qwen3](https://qwen.ai/blog?id=qwen3), same as the previous post. It's a very tiny text generation model that I feel performs very well. Here's huggingface's link to the model: <br>
{{< huggingface model="Qwen/Qwen3-0.6B" >}}

## Setup
Before we start, make sure you installed all the dependencies. Open termianl and type these codes to install depencies:
```sh
pip install torch transformers tokenizers -U
```

Then in terminal, to start python:

```sh
python
```

..Or ipython if you installed jupyter notebook previously:

```sh 
ipython
```

You should see something like this in your terminal, indicating the beginning of a python session:

```sh
Python 3.13.7 | packaged by conda-forge | (main, Sep  3 2025, 14:24:46) [Clang 19.1.7 ] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>>
```
### depencies
Now import dependencies:
```python
import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM
```

<div class="aSeparator"></div>
<i>Sidenote:<br>
As of November 2025, Pytorch/ Apple still haven't fully fixed the <a href="https://github.com/pytorch/pytorch/issues/91368">memory leak</a> issue for Apple Silicon devices. As a result, running models with pytorch may gets slower and slower over time, or freeze compeletly. If you were running a Macbook purchased after 2020, I'd recomment manually set pytorch device as 'cpu'. Skip this code if you were confident that this memory leak won't happen:</i>

```python
torch.set_default_device('cpu')
```
<div class="aSeparator"></div>

### model
Finally, load the model:
```python
DEVICE = torch.get_default_device()
CHECKPOINT = "Qwen/Qwen3-0.6B"
print("Using device:", DEVICE)
model = Qwen3ForCausalLM.from_pretrained(CHECKPOINT)
model.requires_grad_(False)
```
<br>
You are now all set.

### example input text

Let's start with an example text input:

```python
example = "Hi"
```

## Step-by-step Text Generation
### Tokenisation

Our firtst problem is that our input (```example```), is ```str```. However, our ```model``` does not accept ```str``` as input. If you just pass our input to the model, you would get this TypeError:
```python
>>> model(example)
TypeError: embedding(): argument 'indices' (position 2) must be Tensor, not str
```

In order to do text-generation, one first need to to convert inputs into machine-readable formats. This conversion process is called [tokenisation](https://arxiv.org/html/2407.11606v3). Tokenisation is not the main focus of today's post, but as a quick ELI5, you can functionally see tokenisers as glorified look-up tables that converts texts into indices. ```Model``` reads these indicies, through it's own internal lookup table, convert them into text matrices.

It works like this:
{{< mermaid >}}
flowchart LR
    ipt(["input"]) <--> tkn["Tokenizer"]
    tkn <--> mdl["Model"]
    classDef default fill: transparent, bg-color: transparent
{{< /mermaid >}}

For example:
{{< mermaid >}}
flowchart LR
    ipt(["text: 'cat sits'"]) <--tokenizer--> tkn["tokens: [[8, 10]]"]
    tkn --> mdl["Model"]
    classDef default fill: transparent, bg-color: transparent
{{< /mermaid >}}

For huggingface's ```transformers``` library, tokenisation is handeled by ```Tokenizer``` objects. Every pre-trained text models in ```transformers``` library all come with their own paired tokenizers. This is the one used by our model:

```python
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
```

To tokenize our text:
```python
tokens = tokenizer(example, return_tensors='pt')
print(tokens)
print(tokens['input_ids'].shape)
```

You would see our tokenized text input:
```python
{'input_ids': tensor([[13048]]), 'attention_mask': tensor([[1]])}
torch.Size([1, 1])
```

### The Transformer
**Base model & the task head**
<div class="aSeparator"></div>

Models in the ```transformers``` library are assembed by two parts: the transformer itself,  task-specific 'head'. Let's have a look at our model architecture by running ```print(model)```:

```python
>>> print(model)
Qwen3ForCausalLM(
  (model): Qwen3Model(
    (embed_tokens): Embedding(151936, 1024)
    (layers): ModuleList(
      (0-27): 28 x Qwen3DecoderLayer(
        (self_attn): Qwen3Attention(
          (q_proj): Linear(in_features=1024, out_features=2048, bias=False)
          (k_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (v_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (o_proj): Linear(in_features=2048, out_features=1024, bias=False)
          (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
          (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
        )
        (mlp): Qwen3MLP(
          (gate_proj): Linear(in_features=1024, out_features=3072, bias=False)
          (up_proj): Linear(in_features=1024, out_features=3072, bias=False)
          (down_proj): Linear(in_features=3072, out_features=1024, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
        (post_attention_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
      )
    )
    (norm): Qwen3RMSNorm((1024,), eps=1e-06)
    (rotary_emb): Qwen3RotaryEmbedding()
  )
  (lm_head): Linear(in_features=1024, out_features=151936, bias=False)
)
```

In output above, our transformer base model is the one called  ```(model): Qwen3Model``` at the very beginning, the task-head is the ```lm_head: Linear``` at the very end. Let's separate them apart:
```python
base_model = model.model
task_head = model.lm_head
```

```print(base_model)```  shall give you the output save as above except the 'lm_head' bits.

**Running the transformer**
<div class="aSeparator"></div>

To do text generation, we first pass the inputs to the ```base_model```:
```python
output = base_model(**tokens)
print(output)
```

Which would give you this output:
```python
BaseModelOutputWithPast(last_hidden_state=tensor([[[ 7.4006, 29.0470, -0.1732,  ..., -1.2643,  1.1580,  1.1260]]]), past_key_values=DynamicCache(layers=[DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer]), hidden_states=None, attentions=None)
```

Output of the transformer ```base_model``` is the ```last_hidden_state``` attribute of the ```BaseModelOutputWithPast``` object:
```python
last_hidden_state = output.last_hidden_state
print(last_hidden_state.shape)
```

Output:
```python
torch.Size([1, 1, 1024])
```

### The Task-Head



<div class="aSeparator"></div>
<div class="aSeparator"></div>
<div class="aSeparator"></div>

## Tokenisation
**Quick sidenote on tokenizer**<br>
On top of the transformer models themselves, we also need some helper objects in order to convert inputs/ outputs between human-readable and machine-readable formats: our inputs and desirerd outputs are both texts, however models only accepts matrices as their inputs and would only output matrices. This conversation process is called [tokenisation](https://arxiv.org/html/2407.11606v3). For huggingface transformers library, tokenisation is handeled by obejcts called ```Tokenizer```. Tokenisation is very important process for linguistic research, and there are a lot of different methods to tokenise texts. <br>

Tokenisation is not the main focus of today's post, but as a quick ELI5, you can functionally see tokenisers as glorified look-up tables that converts texts into indices. ```Model``` reads these indicies, through it's own internal lookup table, convert them into text matrices. Pre-trained text models from ```transformers``` library all come with their own paired tokenizers.<br>
It works like this:
{{< mermaid >}}
flowchart LR
    ipt(["input"]) <--> tkn["Tokenizer"]
    tkn <--> mdl["Model"]
    classDef default fill: transparent, bg-color: transparent
{{< /mermaid >}}

For example:
{{< mermaid >}}
flowchart LR
    ipt(["text: 'cat sits'"]) <--tokenizer--> tkn["index: [[8, 10]]"]
    tkn <--model--> mdl["matrix: <br>[[0.3, 0.2, 0.5], <br>[0.1, 0.7, 0.3]]"]
    classDef default fill: transparent, bg-color: transparent
{{< /mermaid >}}

**converter**<br>
In order to test out the model, we'll first need to prepare a function that converts inputs & outputs back and forth. Copy this function and paste into python:
```python
def model_converter(tokenizer, inputs: Union[str, ModelOutput, torch.Tensor]) -> Union[str, BatchEncoding]:
    """Converts between text inputs and token indicies

    Args:
        tokenizer (PreTrainedTokenizerFast): the tokenizer to use
        inputs (Union[str, ModelOutput, torch.Tensor]): text inputs or model outputs
    """
    # text -> model inputs
    if isinstance(inputs, str):
        message = [
            {"role": "user", "content": inputs}
        ]
        tokens = tokenizer.apply_chat_template(
            message, 
            tokenize=True, 
            return_dict=True, 
            enable_thinking=False,
            add_generation_prompt=True,
            return_tensors='pt'
            )
        return tokens
    
    # in this demo, we will enconter several different types of model outputs...
    # outputs from model.generate()
    elif isinstance(inputs, torch.Tensor):
        if inputs.ndim==1:
            return tokenizer.decode(inputs)
        elif inputs.ndim > 1 and inputs.shape[0]==1:
            return model_converter(tokenizer, inputs[0])    
        else:
            return tokenizer.batch_decode(inputs)
    
    # outputs from model.forward
    elif isinstance(inputs, ModelOutput):
        if 'logits' in inputs:
            sequences = inputs.logits.topk(1, dim=-1).view(1, -1)
            return model_converter(tokenizer, sequences)
    
    raise TypeError(type(inputs))
```

<br>
Copy these codes to initialise the tokenizer:

```python
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
convert:Callable = partial(model_converter, tokenizer)
```

### Model
Copy these codes to initialise the model:
```python
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT)
model.generation_config.max_new_tokens = 128
model.requires_grad_(False)
```
You are now all set.

## Running the model

Before dive into the text generation mechaism, let's try some text generation first.
Copy this example text or use whatever other texts you'd like to have a try on:
```python
example = "Why are you running?"
```

**Tokenisation**<br>
First, we'll need to convert input text into model-redable form:
```python
tokens = convert(example)
```
Running ```print(tokens)``` would print out something looks like a dictionary with keys like ```input_ids``` and ```attention_mask```. ```input_ids``` are our main interest, these are our formatted input that gets converted into model-redable matrix. Using ```convert```, we can convert the tokens back into text:

```python
print(convert(tokens['input_ids']))
```
...which would give you something like:

```python
<|im_start|>user
Why are you running?<|im_end|>
<|im_start|>assistant
<think>

</think>
```

 "<|im_start|>user", "<|im_start|>assistant", "&lt;think&gt;" and "&lt;/think&gt;" are ```extra special tokens``` that manually amended by the ```model_converter``` in order to format our inputs into a format that the model was orinially trained with. Formatting input this way would help the model generating better results. Different tokenizers have different set of special tokens. For Qwen3 used here, you can view all special tokens by calling tokenizer.get_added_vocab() to see a list of all theset tokens.<br>

**Text Generation**<br>
Use ```model.generate``` to run text generation:
```python
outputs = model.generate(**tokens)
outputs
```
You would see that our model returned antoher matrix of token indicies:

```python
tensor([[151644,    872,    198,  10234,    525,    498,   4303,     30, 151645,
            198, 151644,  77091,    198, 151667,    271, 151668,    271,     40,
           2776,    537,   4303,     13,    358,   2776,   1101,   1588,    311,
           1492,    498,    448,    697,   4755,     13,   6771,    752,   1414,
           1246,    358,    646,   7789,    498,      0, 151645]])
```

Use ```convert``` to convert it back to text:

```python
print(convert(outputs))
```

text generation is a randomised process, therefore your result may be different from mine:
```python
<|im_start|>user
Why are you running?<|im_end|>
<|im_start|>assistant
<think>

</think>

I'm not running. I'm just here to help you with your questions. Let me know how I can assist you!<|im_end|>
```

## Text generation, one word at a time
Instead of generating the entire sentence at once, model generates texts one token at a time. We'll first take a look at how model generates one word from the input text, and look at the entire text generation process in the next section. For a quick recap on the model architecture, please refer to the previous post: [here's the link](https://yuyiheng.cc/posts/transformers-pt-1/#the-transformer-itself). <br>

Text models in the ```transformers``` library are assembed by two parts, the ```XXXModel``` itself and task-specific ```xx_head```. The ```XXXModel``` refers to the **transformer model** itself, and ```xx_head``` is a converter that converts transformer outputs according to different tasks, usually called **task head**.<br>

Let's have a look at our model by running ```print(model)```:

```python
>>> print(model)
Qwen3ForCausalLM(
  (model): Qwen3Model(
    (embed_tokens): Embedding(151936, 1024)
    (layers): ModuleList(
      (0-27): 28 x Qwen3DecoderLayer(
        (self_attn): Qwen3Attention(
          (q_proj): Linear(in_features=1024, out_features=2048, bias=False)
          (k_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (v_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (o_proj): Linear(in_features=2048, out_features=1024, bias=False)
          (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
          (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
        )
        (mlp): Qwen3MLP(
          (gate_proj): Linear(in_features=1024, out_features=3072, bias=False)
          (up_proj): Linear(in_features=1024, out_features=3072, bias=False)
          (down_proj): Linear(in_features=3072, out_features=1024, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
        (post_attention_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
      )
    )
    (norm): Qwen3RMSNorm((1024,), eps=1e-06)
    (rotary_emb): Qwen3RotaryEmbedding()
  )
  (lm_head): Linear(in_features=1024, out_features=151936, bias=False)
)
```

In the outputs shown above, ```(model)``` is our transformer, and ```(lm_head)``` is our text-generation head. To demonstrate, let's use the example mentioned in the previous section:
```python
# convert texts into tokens
example = "Why are you running?"
tokens = convert(example)
# pass tokens to the transformer
transformed = model.model.forward(**tokens)
```

The raw output from ```transformers``` models contains not only the final output, but also computation results inbetween. These inbetween data are very useful for anyone who's interested in figuring out behaviours of their model. For now we are only interested in the final output, ```last_hidden_state```. Let's take a look:
```python
print('Shape of the input matrix:', tokens['input_ids'].shape)
print('Shape of the output matrix:', transformed.last_hidden_state.shape)
print('Output:\n', transformed.last_hidden_state)
```
Output:
```python
Shape of the input matrix: torch.Size([1, 17])
Shape of the output matrix: torch.Size([1, 17, 1024])
Output:
 tensor([[[  5.1596,  19.8593,  -0.2142,  ...,  -0.9417,   0.8061,   0.8258],
         [  1.6308,  26.9123,  -1.1482,  ...,  -3.7286,  -3.3529,  -0.7485],
         [ -0.9393,   7.0810,  -1.4507,  ...,  -0.8947,  -1.6471,  -2.8123],
         ...,
         [  4.9579, -29.7081,   0.0470,  ...,   0.9277,  -2.9034,   0.9229],
         [  3.7464,  -7.0461,   0.1408,  ...,   4.0884,  -1.4918,  -1.3324],
         [ -0.5219,  12.8126,  -1.1671,  ...,   2.8096,   0.8872,  -0.3728]]])
```

We then pass the outputs of our transformer to the task head, ```lm_hed```:
```python
logits = model.lm_head(transformed.last_hidden_state)
print("Shape of the final output matrix:", logits.shape)
```

Output:
```python
Shape of the final output: torch.Size([1, 17, 151936])
```
As you can see, through the computation, our text inputs gets converted into a matrix of indicies of size ```1 x 17```, then gets transformed into a matrix of size ```1 x 17 x 1024```, and then gets converted into a (very big!) matrix of size ```1 x 17 x 151,936```:

{{< mermaid >}}
flowchart TB
    text(["text"])--tokenizer-->ipt["matrix: 1 x 17"] --"model"--> tfm["matrix: 1 x 17 x 1024"] --"lm_head"-->output["matrix: 1 x 17 x 151,936"]
    classDef default fill: transparent, bg-color: transparent
{{< /mermaid >}}



TODO: MERGE PREVIOUS SECTION (THE DEMO TEXT GENERATION) WITH THE CURRENT SECTION
TODO: RE-WRITE THE convert FUNCTION TO FIT MY CURRENT NARRATIVE!!!