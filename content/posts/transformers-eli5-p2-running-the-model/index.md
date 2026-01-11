---
title: "ELI5 Transformers  - Running the model"
date: 2025-12-27
publishDate: 2025-12-27

summary: "(you just need format your inputs)"
font_family: "Monospace"
tags: ["huggingface", "ELI5",]
topics: "transformers"
featureimage: https://imgs.xkcd.com/comics/transformers.png
draft: false
---

{{< katex >}}

*This is part of the ELI5 transformers series, however you do not need to read any previous posts in order to follow the current one. Link to the previous article can be found [in this link](https://yuyiheng.cc/posts/transformers-pt-1/)*

Having a trained transformer model alone is not enough to practically solve most tasks: you also need infrastructure that converts inputs and outputs between human- and model-readable formats. From experience, much of the work in AI is actually on building a maintainable, production-ready ecosystem that manages the model rather than on the model itself. The majority of the cost (both in time and resources) for running and maintaining AI comes from this supporting infrastructure.

Different tasks (i.e., text-to-voice, audio transcription, text-generation, paragraph summrisation, etc.,) would require different infrastructures, of cource, but the general idea behind running these tasks are the same.

This post mainly focuses on a general introduction on running a already trained model, with main focus on the follow 3 sections:

1. **How to get a pretrained model**
    - setup virtual environment for testing ideas
    - download model from checkpoint

2. **How to run the model**
    - converting inputs into model-readable format
    - passing model-readable inputs to the model
    - and converting model outputs back to human-readable format

3. **Understand what happens when running the model**
    - basic model structure: ```base model``` and ```task head```
    - A step-by-step walkthrough of matrix transformation during the text generation
***
## Getting a trained model
![transformers!](https://imgs.xkcd.com/comics/transformers.png "relevant XKCD")

### Preparing the environment

First, install dependencies and start an interactive Python session. In the terminal, create a virtual environment so the work in this post doesn't affect other projects:

```Fish
python -m venv transformers 
```

If you were using <highlight>windows</highlight>, type these in your terminal:

```powershell
transformers\Scripts\activate
```

For <highlight>mac</highlight> and <highlight>linux</highlight>, run:

```bash
source transformers/bin/activate
```

Then install dependencies and start Python:

```Fish
pip install torch tokenizers -U
pip install transformers --pre
python
```

All set.

### Download the model

The model in this demo is ```Qwen3```, a small text-generation model that performs surprisingly well.
{{< huggingface model="Qwen/Qwen3-0.6B" >}}

Import basic dependencies:

```python
import pprint
import torch
from transformers import AutoModelForCausalLM
```

> [!TIP]-for Mac users
> As of November 2025, Pytorch/ Apple still haven't fully fixed the <a href="https://github.com/pytorch/pytorch/issues/91368">memory leak</a> issue for Apple Silicon devices. As a result, running models with pytorch may gets slower and slower over time, or freeze compeletly. If you were running a Macbook purchased after 2020, I'd recommend manually set pytorch device as 'cpu'.<br>
> Skip this if you were confident that this memory leak won't happen:
> ```python
> torch.set_default_device('cpu')
> ```

We then download our model from the [huggingface hub](https://huggingface.co/Qwen/Qwen3-0.6B):

```python
device = torch.get_default_device()
checkpoint = r"Qwen/Qwen3-0.6B"  # link to the model
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
model.requires_grad_(False)  # setting our model in inference mode
```

## Running the model

### Processors

Now we try some inputs:

```python
>>> model.generate("this is a test input")
...
   2546 if "inputs_tensor" in inspect.signature(decoding_method).parameters.keys():
   2547     generation_mode_kwargs["inputs_tensor"] = inputs_tensor
-> 2548 batch_size = inputs_tensor.shape[0]
   2550 device = inputs_tensor.device
   2551 self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

AttributeError: 'str' object has no attribute 'shape'
```

```help(model.generate)``` explains that our model expects```torch.Tensor```, instead of ```str```:

```python
Help on method generate in module transformers.generation.utils:
...
Parameters:
        inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
            The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
            method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
            should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
            `input_ids`, `input_values`, `input_features`, or `pixel_values`.
...
```

In order to run the model, we need to convert text into `torch.Tensor` objects.

Almost all transformer models would require one to first convert inputs into model-readable formats. Ways of data preprocessing changes greatly depending on our desired inputs and desired output. Taking image generation for example, our inputs would be texts ('generate me a cat pictrure') and our outputs would be images ('cat-picture.png'). There are a lot of different processors, depending on your task. Here are some examples:

|Type|Example Processor|
|--|--|
|text|[huggingface tokenizers](https://huggingface.co/docs/tokenizers/index), [nltk](https://www.nltk.org/api/nltk.tokenize.html), [spacy](https://spacy.io/usage), [tiktoken](https://github.com/openai/tiktoken)|
|image|[torch vision](https://docs.pytorch.org/vision/stable/index.html), [huggingface](https://huggingface.co/docs/transformers/v5.0.0rc1/en/main_classes/image_processor#transformers.BaseImageProcessorFast), [scikit-image](https://scikit-image.org/)|
|video|[torchcodec](https://meta-pytorch.org/torchcodec/stable/generated_examples/), [llava](https://huggingface.co/docs/transformers/model_doc/video_llava), [whisper](https://pypi.org/project/openai-whisper/)|
|audio|[torchaudio](https://docs.pytorch.org/audio/stable/index.html)|
|reinforcement learning|[torchrl](https://docs.pytorch.org/rl/stable/index.html), [Gymnasium](https://github.com/Farama-Foundation/Gymnasium), [brax](https://github.com/google/brax)|
|mixed|[huggingface's Processor](https://huggingface.co/docs/transformers/v5.0.0rc1/en/processors#processor-classes), [vllm](https://docs.vllm.ai/en/v0.6.6/design/multimodal/multimodal_index.html)|
||

Despite all these variations, all these processors are doing essentially the same task: converting inputs into model-readable forms. However, it is worth noting that:

- For most cases, processors are model-specific, that is, a model would only work if it was paired with the processor that used in model training. For example, a GPT2 model trained with GPT2 tokenizer would not function propery with a BERT tokenizer.
- Modifying the processor usually means we would need to modify the model accordingly. This does not always mean we need to re-train the already-trained model, but one should always **check** if this was needed.

#### Tokenizer

Let's go back to our text generation example. Raw text data is processed via [tokenisation](https://arxiv.org/html/2407.11606v3). Technic details behind this process very complicated, you can check out a more detailed [guide](https://huggingface.co/docs/transformers/tokenizer_summary). When it comes to production, you can simply think tokenisers as some sort of very glorified look-up tables that converts texts into indices that our ```model``` recognises:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
example = "Hi!"
token = tokenizer(example, return_tensors="pt")
pprint.pprint(token)
```

This converts our result to model-ready format:

```python
{'input_ids': tensor([[13048,     0]]), 'attention_mask': tensor([[1, 1]])}
```

Let's now try with model generation:

```python
result = model.generate(**token, max_new_tokens=20)  # set max-length to stop model generating forever
print(result)
```

Output:

```python
tensor([[13048,     0,   358,  1184,  1492,   448,   419,  3491,    13,   576,
          3491,  2727,    25,   362,   220,    16,    15,    15,    15, 20972,
          3745,   374]])
```

And use our ```tokenizer``` to convert result back to human readable form:

```python
print(tokenizer.decode(result))
```

Output:
```python
>>> print(tokenizer.decode(result))
...
['Hi! I need help with this problem. The problem says: A 1000 kg box is']
```
#### Chat template
Although our model indeed generated something meaningful, this output isn't really what we've expected! This is because our model, just like most of other chat-based text generation models, expects inputs in a very **specific** format: **chats**:

```python
text = "Hi!"
message = [{"role":"user", "content": text}]  # format our input as chat message
message = tokenizer.apply_chat_template(
  message,
  tokenize=False,  # to see our formatted text
  enable_thinking=False,  # disable thinking mode
  add_generation_prompt=True
)
print(message)
```

As seen in the output, our input text is now formatted as chat that message that model would recognise:

```python
...
<|im_start|>user
Hi!<|im_end|>
<|im_start|>assistant
<think>

</think>
>>>
```

Now we tokenize the message and run the text generation

```python
token = tokenizer(message, return_tensors="pt")
result = model.generate(**token, max_new_tokens=20)
print(tokenizer.decode(result)[0])
```

Output:

```python
<|im_start|>user
Hi!<|im_end|>
<|im_start|>assistant
<think>

</think>

Hello! How can I assist you today?<|im_end|>
>>>
```

In addition to the text generated by the model, our raw output also contains input text. We can get rid of them:

```python
generated_tokens = result[0, token['input_ids'].shape[1]:]
generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(generated_text)
```

We finally get our decoded output:

```python
Hello! How can I assist you today?
>>>
```

### Running text generation

Putting everything together, here's our text-generation function:

```python
from functools import partial
def run_text_generation(
  model, tokenizer, input_text, enable_thinking=False, max_new_tokens=256
) -> None:
  """Run text generation and print out the generate text"""
  # format as chat
  message = [{"role":"user", "content": input_text}]  
  # tokenize
  tokens = tokenizer.apply_chat_template(
    message,
    enable_thinking=enable_thinking,
    add_generation_prompt=True,
    return_tensors="pt", 
    )
  # run text generation
  result = model.generate(**tokens, max_new_tokens=max_new_tokens)
  # decode
  generated_tokens = result[0, tokens['input_ids'].shape[1]:]
  generated_text = tokenizer.decode(generated_tokens)
  print(generated_text)

generate = partial(run_text_generation, model, tokenizer)
```

To run our function:

```python
generate("how are you?", enable_thinking=True)  # Let's try with thinking mode!
```

Output:

```text
<think>
Okay, the user asked, "how are you?" I need to respond appropriately. Let me start by acknowledging their question. I should be friendly and open. Maybe say something like, "Hi! I'm here to help. How are you feeling today?"

I should keep the tone positive and offer assistance. Maybe mention that I'm available to help with anything. Also, make sure the response is concise but covers the key points. Let me check if there's any specific context I should consider, but since the user didn't mention anything else, I can proceed with a general response. Alright, that should work.
</think>

Hi! I'm here to help. How are you feeling today? 😊<|im_end|>
>>>
```

### Putting everything together

To recap what's covered in the previous demo, the full workflow of running a transformer model would be:

{{< timeline >}}

  {{< timelineItem icon="code" header="Stage 1" badge="Pre-process" >}}
    The input gets converted into model-readable form.
  {{< /timelineItem >}}

  {{< timelineItem icon="code" header="Stage 2" badge="Transformer" >}}
    The pre-processed inputs feed into the model.
  {{< /timelineItem >}}

  {{< timelineItem icon="code" header="Stage 3" badge="Post-process" >}}
    Model outputs is converted back to human readable form.
  {{< /timelineItem >}}

{{< /timeline >}}

On a side note, this workflow is suprising similar to the model architecture of a transformer model, as mentioned in [the previous post](https://yuyiheng.cc/posts/transformers-pt-1/#transformer-model-architecture):

- Stage 1 converts inputs indices into vectors so that it's model-readable (```Embedding```).
- Stage 2 computes the vector representation of the input. This is the main focus of the [previous post](https://yuyiheng.cc/posts/transformers-pt-1/) (```Transformer```).
- Stage 3 converts raw inputs to task-specific output (```Model output```).

## Model Inference

Transformers are, just like all other neural networks, functions that handles matrix computations. Considering it's size, it is basically impossible to trace every internal calculation steps. On the other hand, understanding how inputs are converted into the outputs can be crucial when it comes to model development and debugging. Instead of looking at how raw numbers gets changed over the calculation steps, looking at how the shape of input matrix gets converted during model inference would be much more helpful and much more informative.

Let's use text-generation as our example. We will be using the same [QWEN3](https://huggingface.co/Qwen/Qwen3-0.6B) model & tokenizer used in the previous section.

Getting the model & the tokenizer:

```python
import pprint
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.get_default_device()
checkpoint = r"Qwen/Qwen3-0.6B"  # link to the model checkpoint
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
model.requires_grad_(False)  # setting our model in inference mode
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

### Task Head

All transformer models can be conceptually divided into  two parts: a ```base_model``` that computs inputs and produces raw logits, and a ```task_head``` that process the raw logits into task-specific outputs:

```python
>>> base_model = model.model
... task_head = model.lm_head
...
... print("base model:", base_model.__class__.__name__)
... print("task head:", task_head)
...
base model: Qwen3Model
task head: Linear(in_features=1024, out_features=151936, bias=False)
>>>
```

### Text generation, step-by-step

Let's go back to our previous exmaple and have a look at how matrices are transformed throughout the entire process.

We first create an text input:

```python
message = [{'role': 'user', 'content': 'how are you?'}]
tokens = tokenizer.apply_chat_template(
  message, 
  add_generation_prompt=True, 
  enable_thinking=False,
  return_tensors="pt"
  )
inputs = tokens["input_ids"]
print("length of formatted text:", len(message))
print("shape of converted tokens:", inputs.shape)
```

We can see that through the tokenizer, our text input was converted from a list of size \(1\) into a matrix of size \(1 \times 16\), as:

**batch_index** \(\times\) **token_indices**

The **batch_index** of **1** here means the model is processing 1 single text entry; where the **token_indices** of **16** means there are 16 tokens in our text input:

```python
...
length of formatted text: 1
shape of converted tokens: torch.Size([1, 16])
>>>
```

Let's first take a look at what our text gets transformed into:

```python
>>> converted_tokens = tokenizer.convert_ids_to_tokens(inputs[0])
>>> pairs = list(zip(inputs[0].tolist(), converted_tokens))
>>> pprint.pprint(pairs)
...
[(151644, '<|im_start|>'),
 (872, 'user'),
 (198, 'Ċ'),
 (5158, 'how'),
 (525, 'Ġare'),
 (498, 'Ġyou'),
 (30, '?'),
 (151645, '<|im_end|>'),
 (198, 'Ċ'),
 (151644, '<|im_start|>'),
 (77091, 'assistant'),
 (198, 'Ċ'),
 (151667, '<think>'),
 (271, 'ĊĊ'),
 (151668, '</think>'),
 (271, 'ĊĊ')]
>>>
```

> For our tokenizer, white spaces are represented by symbol **Ġ** and **Ċ**.

As we can see from the output, our input gets formatted into a chats, and then conveted into a matrix of integers, with each entry represent the ```token_id``` of the corresponding text string. If you ```print(inputs)```, you would see the actual inputs that model receives:

```python
>>> print(inputs)
tensor([[151644,    872,    198,   5158,    525,    498,     30, 151645,    198,
         151644,  77091,    198, 151667,    271, 151668,    271]])
>>>
```

***
We then pass our the tokens to the base model:

```python
output = base_model(**tokens)
last_hidden_state = output.last_hidden_state
print("shape of base_model output:", last_hidden_state.shape)
```

Result:

```python
...
shape of base_model output: torch.Size([1, 16, 1024])
```

Through the ```base_model```, our input tokens of size \(1 \times 16\) gets transformed into a matrix of size \(1 \times 16 \times 1024 \), formatted as:

**batch_index** \(\times\) **token_id** \(\times\) **token_representation**

This output from our ```base_model``` can be seen as the final transformed matrix representation of our original input data:

```python
>>> print(last_hidden_state)
tensor([[[  5.0938,  18.7500,  -0.2129,  ...,  -0.9297,   0.7852,   0.8203],
         [  1.7891,  26.6250,  -1.1484,  ...,  -3.8906,  -3.3906,  -0.8125],
         [ -1.0000,   7.3125,  -1.4531,  ...,  -0.8672,  -1.6797,  -2.7969],
         ...,
         [  3.4062, -32.2500,   0.1260,  ...,   1.3906,  -3.0938,   1.4844],
         [  3.7656,  -7.3750,   0.2070,  ...,   3.7500,  -1.4375,  -1.3984],
         [ -1.4531,  34.0000,  -0.6484,  ...,  -0.7188,   1.3047,   1.0625]]],
       dtype=torch.bfloat16)
>>>
```

***
We then pass the output of  ```base_model``` to the ```task_head```:

```python
model_output = task_head(last_hidden_state)
print("shape of the final model output:", model_output.shape)
print("vocabulary size:", len(tokenizer.vocab))
```

Result:

```python
...
shape of the final model output: torch.Size([1, 16, 151936])
vocabulary size: 151669
```

This output matrix of \( 1 \times 16 \times 151936\) is:

**batch_size** \( \times \) **token_index**  \( \times \) **token_score**

Where **token_score** refers to the *score* for the next token index (next text) given all previous text. For example, **token_score** at [ 0, **7**, : ] in our output of \( 1 \times 16 \times 151936\) matrix refers to the *score* of **7th** token given previous token **0-6**.

To get our 17th token, we only need the last sequence of our ```model_output```:

```python
next_token_score = model_output[:, -1, :]
print("shape of next_token_score:", next_token_score.shape)
print("value of next_token_score:", next_token_score)
```

Result:

```python
...
shape of next_token_score: torch.Size([1, 151936])
value of next_token_score: tensor([[14.1250, 12.9375,  7.6875,  ...,  4.3750,  4.3750,  4.3750]],
       dtype=torch.bfloat16)
```

To convert score to probablities, although not necessary, use ```softmax```:

```python
>>> next_token_probability = torch.nn.functional.softmax(next_token_score, dim=-1)
>>> print(next_token_probability)
tensor([[9.7603e-07, 2.9802e-07, 1.5643e-09,  ..., 5.7071e-11, 5.7071e-11,
         5.7071e-11]], dtype=torch.bfloat16)
```

***
Finally, we need to use ```tokenizer``` to convert model output to human-redable format. We use tokens with highest ```next_token_score``` as the next token for text generation:

```python
top_score, top_token_index = next_token_score.topk(1, dim=-1)
print("top 1 score:", top_score)
print("top 1 next token index:", top_token_index)
print("shape of output:", top_token_index.shape)
```

Our ```top_token_index``` matrix has shape \( 1 \times 1 \):

```python
...
top 1 score: tensor([[27.7500]], dtype=torch.bfloat16)
top 1 next token index: tensor([[9707]])
shape of output: torch.Size([1, 1])
```

That is **batch_index** \( \times \) **top1_token_index**

***
We use ```tokenizer```, the processor for text transformers, to decode ```top_token_index``` back to text:

```python
decoded_next_token = tokenizer.decode(top_token_index)
print(decoded_next_token)
```

And here it is:

```python
...
['Hello']
```

***

### Recap: transformation journey

The demo in the section above shows how matrix is transformed step-by-step through the whole model inference process, here's a recap of how the shape of our input data changes throughout the computation process:

{{< mermaid >}}

---
config:
  layout: dagre
  theme: redux
  look: neo
---
flowchart TB
 subgraph Model["<br>"]
        n4["last_hidden_state<br>size: [1, 16, 1024]"]
        n5["next_token_score<br>size: [1, 16, 151936]"]
  end
    n1@{ label: "input: 'how are you?'" } --> n2(["chat message<br>size: [1]"])
    n2 -- tokenizer --> n3["tokens<br>size: [1, 16]"]
    n3 -- base model --> n4
    n4 -- task head --> n5
    n5 -- top 1 --> n6(["top_token_index<br>size:[1, 1]"])
    n6 -- tokenizer --> n7(["decoded_next_token<br>size: [1]"])
    n7 --> n8@{ label: "next token: 'Hello'" }

    n4@{ shape: braces}
    n5@{ shape: braces}
    n1@{ shape: rect}
    n3@{ shape: terminal}
    n8@{ shape: rect}
    classDef default fill: transparent, bg-color: transparent
    style n1 stroke-width:4px, stroke-dasharray: 0
    style n8 stroke-width:4px, stroke-dasharray: 0
    style Model fill:transparent

{{< /mermaid >}}
 
## Conclusion

Running transformers boils down to three things: preprocess inputs into model-readable formats, pass them through the model, and post-process outputs back to human-readable forms. Much of the work in machine learning isn't actually about the model—it's about the supporting infrastructure. Processors are as important as the model when it comes to using them. I would say understanding data flow and how to format inputs and interpret outputs would be more than enough to cover 80% of using transformers as productivity tools.

The next post will take a more in-depth look at text-generation, and starting to divert the main focus from inference to model training.

## Some further reading

Here are some external reading materials that I found very useful when it comes to learning model inference:

1. [pytorch's intro on neural networks](https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html).
2. [pytorch's guide on torch.nn library](https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html), this covers 90% of the basics you need to know for pytorch models from the ```transformers``` library.
3. [transformer's tutorial on running model pipelines](https://huggingface.co/learn/llm-course/chapter1/3), instead of writing your own functions of parsing the model, ```transformers``` library provides ```Pipeline```s that runs the entire model inference from human-readable inputs to human-readable outputs. The linked tutorial covers the basics for how to use them. The basic structure behind ```Pipeline```s are exactly the same as what described in this post: ```preprocess```, ```model forward```, ```post process```.