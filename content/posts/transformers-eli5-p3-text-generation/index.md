---
title: "ELI5 Transformers - Demystify Text Generation"
date: 2026-01-18
publishDate: 2026-01-18

summary: "(it's glorified sequence classification)"
tags: ["huggingface", "ELI5",]
topics: "transformers"
draft: false
---


{{< katex >}}
*This is part of the ELI5 transformers series, however you do not need to read any previous posts in order to follow the current one. Link to the series can be found [here](https://yuyiheng.cc/topics/transformers/)*

I'm pretty sure you've already heard about something like this before: *'generative LLM is just slightly advanced auto complete'*. Or, something like *'predicting next word given all the previous words'*. Today I would like to invite you think of text generation from a slightly different perspective: <br>

<highlight>Text generation is sequence classification in a for-loop.</highlight>

![always has been](./featured.jpg "always has been")

Rather than building things bottom-up by covering all the basics first, when it comes to learning neural networks (or any programming-related tools in general), I find it significantly easier to start from the big picture and work downward. This post begins by showing how to use a pre-trained text generation model to produce text. It then explains how text generation works, focusing on inputs and outputs from the model: what kind of data model receives, what kind of data model outputs, and do we to interpret the model outputs. And finally, this post will very briefly discuss how text generation process mirrors a 'decision tree' and reinforcement learning.

***

## Preparation

### Setup environment

#### conda (recommend)

[Install miniconda first](https://www.anaconda.com/docs/getting-started/miniconda/install##quickstart-install-instructions) if you haven't already. After installing, open terminal and type:

```Fish
conda create -n transformers python=3.13 -y
```

And then activate the environment you've created:

```fish
conda activate transformers
```

***

#### venv (comes with python)

First, install dependencies and start an interactive Python session. In the terminal, create a virtual environment so the work in this post doesn't affect other projects:

```Fish
python -m venv transformers 
```

If you were using <highlight>windows</highlight>, type these in your terminal:

```powershell
transformers\Scripts\activate
```

For <highlight>mac</highlight> and <highlight>linux</highlight>, run:

```Fish
source transformers/bin/activate
```

***

### Prepare dependencies

Then install dependencies and start Python:

```Fish
pip install rich -U
pip install torch tokenizers -U
pip install transformers --pre
python
```

All set.
***

In python, copy and paste these codes to import dependencies:

```python
import pprint
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
```

> [!tip]-for Mac users
> As of November 2025, Pytorch/ Apple still haven't fully fixed the <a href="https://github.com/pytorch/pytorch/issues/91368">memory leak</a> issue for Apple Silicon devices. As a result, running models with pytorch may gets slower and slower over time, or freeze compeletly. If you were running a Macbook purchased after 2020, I'd recommend manually set pytorch device as 'cpu'.<br>
> Skip this if you were confident that this memory leak won't happen:
>
> ```python
> torch.set_default_device('cpu')
> ```

***

### Getting the model

Same as [previous posts](https://yuyiheng.cc/topics/transformers/), we'll be using the mini-version of **Qwen3** model for today's demo:

{{< huggingface model="Qwen/Qwen3-0.6B" >}}

And to download our model:

```python
device = torch.get_default_device()

checkpoint = r"Qwen/Qwen3-0.6B"  ## link to the model
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
model.requires_grad_(False)  ## turn off training mode for faster inference
model.generation_config.max_new_tokens = 256  ## to make things runs slightly quicker,,
```

You would see a status bar whilst downloading & loading our model:

```python
...
Loading weights: 100%|██| 311/311 [00:00<00:00, 4366.64it/s, Materializing param=model.norm.weight]
The tied weights mapping and config for this model specifies to tie model.embed_tokens.weight to lm_head.weight, but both are present in the checkpoints, so we will NOT tie them. You should update the config with `tie_word_embeddings=False` to silence this warning
```

> You can safely ignore the warning messages, since we aren't going to modify the model. However you will need to handle these warnings when it comes to actual development!

***

## Run text generation

The general workflow of models from ```transformers``` library has three distinct stages: preoprocess, model forward, and postprocess (you can find more details [in the previous post](https://yuyiheng.cc/posts/transformers-pt-2/##putting-everything-together)):

{{< mermaid >}}
flowchart LR
    ipt(["Input"]) -- pre-process --> model
    model -- post-process --> output(["output"])
    classDef default fill: transparent, bg-color: transparent
{{< /mermaid >}}

### Preprocess: preparing example data

We'll be working with an example message for our text generation demo:

```python
example = "how are you?"
```

In order for model to perform optimally, input data needs to be formatted similar to what model was originally trained on. For our **QWEN3** model, it's this specific **chat format**:

```python
message = [{"role":"user", "content": example}]
formatted_example = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
print(formatted_example)
```

Our formatted input texts looks like this:

```python
...
<|im_start|>assistant
<think>

</think>


```

Models from ```transformers``` library do not accepts raw texts as inputs, only integers of token indicies. We need to convert our formatted texts to tokens before passing to our model:

```python
tokens = tokenizer(formatted_example, return_tensors="pt")

## or via tokenizer.apply_chat_template with tokenize=True:
tokens = tokenizer.apply_chat_template(
  message, 
  add_generation_prompt=True, 
  enable_thinking=False,
  tokenize=True,
  return_dict=True, 
  return_tensors="pt",
)

print(tokens) ## {'input_ids': tensor([[151644,  77091,    198, 151667,    271, 151668,    271]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1]])}
```

### Running model.generate()

Text generation models from the ```transformers``` library provides a very easy and simple API for text generation, ```model.generate```:

```python
>>> result = model.generate(**tokens)
... generated = result[0, tokens['input_ids'].shape[1]:]  ## skip input tokens
... print(generated)
tensor([9707, 0, 2585, 525, 498, 3351, 30, 26525, 232, 151645])
```

### Postprocess

Then, we convert raw output back to human-readable format via tokenizer.decode:

```python
>>> tokenizer.decode(generated)
'Hello! How are you today? 😊<|im_end|>'
```

***

## Text generation explained

The text generation via method ```model.generate()``` is essentially running ```model.forward``` over and over till model genrates end of srequence token (as defined in ```model.config.eos_token_id```)

Now that we know the basics of text generation, we can take a look at how text generation works.

### Understand model outputs


#### CausalLMOutputWithPast object


To understand how text generated, we can start with the raw output of our model, and see how raw outputs are converted to human-redable texts in the ```post process``` stage. 

We pass our pre-processed inputs ```tokens['input_ids']``` to our model to get raw model output:

```python
inputs = tokens['input_ids']
output = model.forward(inputs)
pprint.pprint(output)
```

Outputs from our model contains a lot of things, as you can see here:

```python
CausalLMOutputWithPast(
  loss=None, 
  logits=tensor([[[ 3.5938,  3.7812,  3.6562,  ...,  1.6484,  1.6484,  1.6484],
         [ 7.2812,  8.2500,  6.3125,  ...,  0.4141,  0.4141,  0.4141],
         [ 4.7188,  9.6250, 11.5625,  ...,  3.8281,  3.8281,  3.8281],
         ...,
         [ 5.7812, 14.7500,  9.5000,  ...,  1.4531,  1.4531,  1.4531],
         [-4.2500, -3.8750, -7.4688,  ..., -1.1719, -1.1719, -1.1719],
         [14.1250, 12.9375,  7.6875,  ...,  4.3750,  4.3750,  4.3750]]],
       dtype=torch.bfloat16), 
    past_key_values=DynamicCache(layers=[DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer, DynamicLayer]), 
    hidden_states=None, 
    attentions=None
)
```

Here's a quick explaination:

|name|explain|
|--|--|
|**loss**|loss of model performance, used in model training|
|**logits**|the main outputs of our model, this is the main focus of current post|
|**past_key_values**|caching, this is used to speed up text generation|
|**hidden_states**|raw output of the base model ([more detail](https://yuyiheng.cc/posts/transformers-pt-2/##task-head)), <br>returned when *output_hidden_states = ```True```*|
|**attentions**|raw output of each attention layer, <br>returned when *output_attentions = ```True```*|

***

#### Matrix transformation

Let's compare our inputs and outputs:

```python
>>> print("Model inputs:", inputs.shape)
>>> print("Model outputs:", output.logits.shape)
...
Model inputs: torch.Size([1, 16])
Model outputs: torch.Size([1, 16, 151936])
```

We can see that through our model (```model.forward(inputs)```), input of size **1 \(\times\) 16** gets *transformed* into output of size **1 \(\times\) 16 \(\times\) 151936**, shaped like this:

**batch index** \( \times \) **token index** \( \times \) **token score**

The size of last dimension of our model output ```output.logits```,  ```151936``` matches the ```vocab_size``` of our tokenizer:

```python
>>> tokenizer.vocab_size
151643
```

For example, item ```[0, 3, 250]``` in our ```output.logits``` means:

|dimension|value|meaning|
|:--|:--|:--|
|batch index| 0 | 1st row of input (python indexing starts from 0)|
|token index| 3 | 4th token|
|token score| 250 | score for input token in: <br>**1st** row, **4th** input token, **251th** token in the tokenizer vocabulary|

And the value is ```7.5625```:

```python
>>> output.logits[0, 3, 25]
tensor(7.5625, dtype=torch.bfloat16)
```

***

#### Final output

The last dimension of model output ```output.logits``` also matches ```out_features``` of the very last layer of our ```model```:

```python
>>> model.lm_head
Linear(in_features=1024, out_features=151936, bias=False)
```

This ```lm_head``` takes matrix with last dimension of size **1024** (```in_features```) as inputs, and outputs matrix with last dimension of size **151936** (```out_features```), for example, for our 5th token, the scores are:

```python
>>> output.logits[0, 4, :]
tensor([ 2.9062,  4.3438,  2.5469,  ..., -2.0312, -2.0312, -2.0312],
       dtype=torch.bfloat16)
```

Where each score can be seen as **model's confidence score of the next token**. For example, the word '*pub*' would be more likely to be the next word following sentence '*I had some beer from the*',  compared with word '*t-shrit*'. The ```output.logits``` above is just a mathemathical way of describing of this difference in likelihood. To get the token with highest likelihood, use ```topk```:

```python
highest_value, highest_index = output.logits[0, 4, :].topk(1)
previous_token = inputs[0, :5]
previous_text = tokenizer.decode(previous_token)
model_prediction = tokenizer.decode(highest_index[0])
print('Prevous text:', previous_text, sep="\n", end="\n================\n")
print("Next token predicted by the model:", model_prediction, sep="\n")
```

Output:

```python
...
Prevous text:
<|im_start|>user
how are
================
Next token predicted by the model:
 the
>>>
```

So given our input ```formatted_example```, the most likely next token, as predicted by the model, would be:

```python
>>> next_token_score, next_token_index = output.logits[:, -1, :].topk(1)
>>> next_token = tokenizer.decode(next_token_index)
>>> print(next_token)
...
['Hello']
```

***

### Parse raw output

To recap what we've found so far:

1. Output of models from ```transformers``` library (via ```model.forward(inputs)```) contains lots of things for caching/ debugging etc. The raw output is the ```logits``` attribtue of ```ModelOutput``` objects.

2. Raw output (```logits```) for text generation models can be considered as model's confidence score of the next token.

Value of ```logits``` for a given token depends not only on the token itself, but also on all the tokens in front of it. For example:

```python
dog = tokenizer(" dog", return_tensors="pt")
hi_dog = tokenizer("Hi dog", return_tensors="pt")
hi_dog_how_are_you = tokenizer("Hi dog how are you?", return_tensors="pt")
hello_dog = tokenizer("Hello dog", return_tensors="pt")

output_dog = model.forward(**dog).logits
output_hi_dog = model.forward(**hi_dog).logits
output_hi_dog_how_are_you = model.forward(**hi_dog_how_are_you).logits
output_hello = model.forward(**hello_dog).logits

print("Value of ' dog' itself:\n", output_dog[:, 0, :])
print("Value of 'dog' in 'Hi dog':\n", output_hi_dog[:, 1, :])
print("Value of 'dog' in 'Hi dog how are you?':\n", output_hi_dog_how_are_you[:, 1, :])
print("Value of 'dog' in 'Hello dog':\n", output_hello[:, 1, :])
```

We can see that the value of the token “dog” is influenced only by the text that comes **before** it, and not by any text that comes *after* it:

```python
Value of ' dog' itself:
 tensor([[5.0000, 4.5938, 4.0312,  ..., 1.2266, 1.2266, 1.2266]], dtype=torch.bfloat16)
Value of 'dog' in 'Hi dog':
 tensor([[13.9375,  6.8125,  7.0938,  ..., -0.6016, -0.6016, -0.6016]], dtype=torch.bfloat16)
Value of 'dog' in 'Hi dog how are you?':
 tensor([[13.9375,  6.8125,  7.0938,  ..., -0.6016, -0.6016, -0.6016]], dtype=torch.bfloat16)
Value of 'dog' in 'Hello dog':
 tensor([[14.5000,  6.4062,  7.0312,  ..., -0.4531, -0.4531, -0.4531]], dtype=torch.bfloat16)
```

Thus, we can see that the raw logits of a token in an input sequence contains information about the entire equence **up to that token**. Since the logits of a token represents the mathematical representation of that token, one can also say:

For text generation, the selection of next token is based on the *mathematical representation of all previous tokens*. 

***

To demonstrate, let's use the shorter, unformatted text input instead:

```python
example = "Hi, how are you doing today?"
tokens = tokenizer(example, return_tensors="pt")
inputs = tokens['input_ids']
outputs = model.forward(**tokens)
logits = outputs.logits
```

And to make things easier to read, let's put everything in a nice table (via ```rich``` library):

```python
from rich.console import Console
from rich.table import Table

console = Console()
table = Table("previous text", "token with highest score", show_lines=True)

for index in range(inputs.shape[1]):
  input_text = tokenizer.decode(inputs[0, :index+1])
  prediction_scores = logits[0, index]
  top_score, top_token = prediction_scores.topk(1)
  prediction_text = tokenizer.decode(top_token)
  table.add_row(input_text, prediction_text)
console.print(table)
```

And here's the output:

```Fish
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ previous text                ┃ token with highest score ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Hi                           │ Question                 │
├──────────────────────────────┼──────────────────────────┤
│ Hi,                          │  I                       │
├──────────────────────────────┼──────────────────────────┤
│ Hi, how                      │  can                     │
├──────────────────────────────┼──────────────────────────┤
│ Hi, how are                  │  you                     │
├──────────────────────────────┼──────────────────────────┤
│ Hi, how are you              │  doing                   │
├──────────────────────────────┼──────────────────────────┤
│ Hi, how are you doing        │  today                   │
├──────────────────────────────┼──────────────────────────┤
│ Hi, how are you doing today  │ ?                        │
├──────────────────────────────┼──────────────────────────┤
│ Hi, how are you doing today? │  I                       │
└──────────────────────────────┴──────────────────────────┘
```

### It's really just sequence lassification

To further prove this point, we'll use a *text classification* model to do our text generation task.
Here's the code that does exactly this:

```python
from transformers import AutoModelForSequenceClassification, AutoConfig
vocab_size = model.config.vocab_size
model = AutoModelForSequenceClassification.from_pretrained(
  checkpoint,
  num_labels = vocab_size,
  key_mapping = {"lm_head.weight":"score.weight"}
)
```

> [!info]-code explaination 
> ```huggingface```'s ```transformers``` models are written in ```torch```, hence you can use all model loading/ inferencing methods from the ```torch``` library. [Here](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) is a very good tutorial going through naive features of ```torch``` library.<br><br>
> In addition, ```transformers``` also provides a LOT of other utility functions that comes *extremey* handy for training/ managibg/ running neural netrowrk models. This includes some very flexiable ways of loding models from existing checkpoints. Here's a brief explination of what's hapenning in the the code:<br>
>
> - To create a text classification model, we first need to determine how many classes are there to begin with. For example, a typical text classification task would be sentimental analysis, where model predicts if the input sentence were 'positive' or 'negative' (*I love fish!!* vs *I hate fish!!*). In this case we'll have total number of 2 classes (positive/ negative). Here, the number of classes is the size of the total vocabulary from the previous loaded model, ```model.config.vocab_size```.
>
> - Our model is initiated via the ```AutoModelForSequenceClassification.from_pretrained``` method. ```AutoModelFor{task-name}``` automatically finds the correct model class (in our case ```Qwen3Model```) with correct task-head, with extra model configurations by the user.The [```num_labels```, ```key_mapping```] essentiall tells ```AutoModelForSequenceClassification``` to create sequence classification model head, with total of ```num_labels``` classes, using ```lm_head.weight``` parameter stored at the ```checkpoint```.
>
> The result is a ```Qwen3ForSequenceClassification``` model, with final layer (```model.score```) using parameters from ```Qwen3ForCausalLM.lm_head```.

As you can see from here, we will be using sequence classification model for our text generation task:

```python
>>> model
Qwen3ForSequenceClassification(
  (model): Qwen3Model(
    (embed_tokens): Embedding(151936, 1024)
...
```

Sequence classification models from ```transformers``` library, including our ```Qwen3ForSequenceClassification``` model, classifies texts by computing the class scores for the ***the last*** non-padding token, essentially *the mathematical representation of the entire input sequence*.

So, using our classification model, let's do some text generation:

```python
max_new_tokens = 20
example = "How are you?"
formatted_example = encode(example)  ## format as chats
for i in range(max_new_tokens):
  tokens = tokenizer(formatted_example, return_tensors="pt")  ## convert inputs to indicies
  outputs = model.forward(**tokens)  ## run 
  top1_score, top1_index = outputs.logits.topk(1, -1)

  decoded_message = tokenizer.decode(top1_index[0])
  print(decoded_message, end="", flush=True)  # to print it out in real time
  formatted_example = formatted_example + decoded_message
print("\n")
```

And we can see texts generated from our model, one by one in real time:

> [!info]-about flush=True
> What Python's print function does is essentially writing data to a nominated destination. By default, ```print(*some_content)``` writes data to ```sys.stdout```, which is the terminal window/ console. You can control where data goes to by setting keyword argument `file`: ```print('hello', file=open(`some_file.txt`, 'w+'))``` would write 'hello' to 'some_file.txt'.
>
> By default (`flush`=False), to save computation resources, this writing behaviour does not happen in real time. Instead, contents of the data first send to a buffer, and as contents will accumulate till the point where the buffer decides it's time to release them to the destination. This 'releasing' behaviour is controlled by the buffer itself, and would very between different operating systems, destination types, user settings etc. In order to get our generated result in real time, we can set ```flush``` to be true, to manually tell buffer to 'flush' its content into the destination, in our case the terminal window.
>
> In the code above, you probably won't be seeing texts generated in real time if you set ```flush``` to be False. But the code would run significantly faster! This is because writing data takes A LOT of resources, and having a buffer would really speed things up a lot (with the cost of not seeing output in real time)!

```fish
Hello! I'm here to help you with anything you need. How can I assist you today?
```

## Beyond classification

### Decision Tree

Text generation models do not directly output the next token. It produces a score for each token in the vocabulary. These scores are serves as a reference for how likely each token is to come next and is later used to select the next token. Imagine you're on your phone and typed ```'Today'```. The keyboard would prompt you with 2 to 3 words it thinks are likely to be the next words, like ```['the', 'I']```; and after selecting ```'the'```, the keyboard would prompt you another 2 ~ 3 next likely words, like, ```['weather, 'rain', 'dog']```, and so on and on:

{{< mermaid >}}
---
config:
  theme: mc
  layout: dagre
  flowchart:
    padding: 0
    rankSpacing: 90
    nodeSpacing: 90
    diagramPadding: 5
    arrowMarkerAbsolute: none
---
flowchart-elk TB
    A(["Today"]) -- "13.2" --> n1["the"]
    A -- "12.3" --> n2["I"]
    A -- "11.2" --> n4["you"]
    n1 -- "2.3" --> n8["school"]
    n1 -- "7.9" --> n9["weather"]
    n1 -- "4.5" --> n11["sky"]
    n9 -- "20.1" --> n12["feels"]
    n9 -- "15.2" --> n13["changed"]
    n9 -- "0.1" --> n16["beyond"]

    linkStyle 0 stroke:#FF6D00,fill:none
    linkStyle 4 stroke:#FF6D00,fill:none
    linkStyle 6 stroke:#FF6D00,fill:none
    classDef default fill: transparent, bg-color: transparent
{{< /mermaid >}}

The tree above shows the top three words predicted by the model at each generation step. The numbers represent the token scores for the corresponding words, as output by the model. The red lines mark the path of tokens with the highest scores at each generation step. What we see is a tree of many possible words within a forest of all possible branches that can follow our starting word. This branching process lookes like this:

{{< mermaid >}}
---
config:
  theme: redux
  flowchart:
    padding: 0
    rankSpacing: 90
    nodeSpacing: 90
    diagramPadding: 5
    arrowMarkerAbsolute: none
---
flowchart TB
    B["text sequence"] -- "model.forward()" --> n1["next token score for each token in the vocabulary"]
    n1 -- "selects next token" --> n2["generated token"] -- "appends to the end of text sequence" --> B

    B@{ shape: rounded}
    n2@{ shape: rect}

    classDef default fill: transparent, bg-color: transparent
{{< /mermaid >}}


This 'pathing' shares stricking similarities with reforcement learning:

{{< mermaid >}}
---
config:
  theme: redux
  flowchart:
    padding: 0
    rankSpacing: 90
    nodeSpacing: 90
    diagramPadding: 5
    arrowMarkerAbsolute: none
---
flowchart TB
    B["past states"] -- "model.forward()" --> n1["Score for every possible action to take"]
    n1 -- executes action --> n2["next state"]
    n2 L_n2_B_0@-- append to the history --> B

    B@{ shape: rounded}
    n2@{ shape: rect}

    classDef default fill: transparent, bg-color: transparent
{{< /mermaid >}}

> This similarity is not just for using model, but also for model training. Think of chess bots like alphachess: model learns what to do for every possible steps of chess, from opening move to the final checkmate. Just like text generation model where texts are generated from ```model.generation_config.decoder_start_token_id```.

### Generation strategy

Model do not select the next token for us. This selection process is done through external algorithms. In our demo above, we used the token with highest score as the next token. On the other hand, there are also a lot of other different ways to choose the next token:

- we can also randomly *draw* the next token from the list of all tokens, where the *sampling probability* is determined by the model’s output score.

- Or, we do some bit of editing to the model's output score, to prevent some tokens getting selected over and over (like 'the', ). Or, we can only sampling from the top, say, 50 tokens.

- Or we can let model traverse down the entire decision tree untill some end-of-sentence mark (like a period mark) whilst keeping a record of the token scores of the path down the way

[Here](https://huggingface.co/blog/how-to-generate) is a good summary article that covers you the basics.

## Further reading

Here are some other articles/ contents that helped me greatly:

- [Pytorch's tutorial on building text generation by scartch](https://docs.pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html). Although this tutorial uses a different infrastructure, it offers an excellent walkthrough of the text generation process using a much, much simpler and cleaneer example. The whole article is very easy to follow and I found the reading process very enjoyable.

- [Pytorch's tutorial on building a text translation bot](https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html), this tutorial builds an attentional-based text generation model on scartch, and provides an excellent introduction of encoder-decoder architecture. Most of the text generation model nowadays are decoder-only, however, understanding how encoder, as well as encoder-decoder works would helps one tremendenously when it comes to transformers.
