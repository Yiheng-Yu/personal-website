---
title: "ELI5 Transformers (Part 1): Attention Mechanism "
date: 2025-10-31
publishDate: 2025-10-31
draft: false
summary: "(it's glorified linear algebra)"
font_family: "Monospace"
tags: ["AI", "Machine Learning", "ELI5"]
topics: "transformers"
params:
  math: true
---
{{< katex >}}

In this particular post, I would like to do a very brief overview of the transformer model architecture, specifically on the attention mechanism. I won't go metion too much math and there won't be any mathenathical formulas. However, I would assume readers of this silly little post already have some okay-ish background of math/ datascience. (i.e., matrix computations embeddings, tokens, model fitting etc.). I am not going to list out all the implemention details for transformers, since there are a lot of very good materials out there and they are doing fantastic jobs. Instead, in this (maybe series of?) post, İ would like to draw out a general framework on transformers to help one understand the detailed math behind.<br>

Today, I'll very quickly go through some very basics on neural network model, just enough to cover what needed for this post, accompied by demo of transformer model as a proof of concept. In this section, there will be some codes that you can copy and paste into an interactive python session to fiddle around for a bit. And lastly, I'll do a quick sketch on the general architecture of transformers, and a overview of the attention mechasm.<br>

## Model only needs to be useful

In order to make things easier to understand, I would wish to start with an inaccuate premise: we can view neural network models as functions that takes some sort of matrix as inputs, do some sort of matrix computations, and output another matrix as the final result. What makes one neural network different from others is how the computation is carried out. It's like \(y=a \times x^2\) is a different function from \(y=a \times sin(x)\), only that in the case of neural network, both x and y are matrices, and the math is much complicated. When it comes to model training, we are essentially trying to find values gives best fit to the data.<br>

There's an important assumption here: just because model fits the data well does not mean the model describes mechanisms behind the data. For example, we <i>definitely</i> can fit \(y=a \times sin(x) + b\) to a normal distribution data (like distribution of customer spendings in McDonald's), and it's prob going to be a pretty good fit, but this does not mean the sine function has anything to do with explaining the normal distribution. A good model does not always need to be description, a good model just needs to be useful for its purpose.<br>

Transformers are preciesly these kinds of models: they are, surprisingly good at fitting into all sorts of data whilst the math behind the model probably doesn't have much to do with the mechanisms behind. We don't know how transformers works so well for text-based tasks. At least not yet. Originally, transformer was designed as an add-on to the text-processing neural network models in order to tackle with some tricky problems (these problems are not the main forcus of the current blogpost so I'm skipping them, but [here's a good article if you were interested](https://towardsdatascience.com/beautifully-illustrated-nlp-models-from-rnn-to-transformer-80d69faf2109/)). We just happened to discover that transformers alone is good enough to solve these problems, we just need to make the transformers much bigger. So that's where the Aİ bloom started: GPT2 solved issues in GPT1 by simply being 10 times bigger; the most-recently open-sourced [pretrained GPT-Oss](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1), is 200 times bigger than [the previous openpsourced model, GPT2](https://huggingface.co/openai-community/gpt2) <i>(note: GPT-OSS is structurlly different from the original GPT2 but the fundamental ideas are the same.)</i>. There are even speculations suggesting transformer neural network models can be seen as some sort of universal function approximator. That is, it's capacable of 'approximate' other formulas/ functions with certain degree of accuracy, providing the model itself is big enough (['universal approximation theorem'](https://en.wikipedia.org/wiki/Universal_approximation_theorem)). <br>

## Transformer Model Architecture
### Overview
At conceptual level, the general idea behind transformer models are is actually pretty intuitive. We can roughly divide the model calclations into three stages:
{{< timeline >}}

{{< timelineItem icon="code" header="Stage 1" badge="Embedding" subheader="The input gets converted into vectors or matrices." >}}
  The first step starts with creating a mathematical representation of our input data. This process can vary based on different types of inputs. It can simply be some sort of look-up tables (text embedding), some matrix transformations of the raw inputs (convolution) etc.
{{< /timelineItem >}}

{{< timelineItem icon="code" header="Stage 2" badge="Transformer" subheader="The raw output from Stage 1 feeds into the multiple different attention layers." >}}
  Mathematically, each attention layer is doing very much the same mathematical operation, with each layer having its own sets of parameters. Each layer takes a matrix as an input, and outputs another matrix to pass onto the next layer. This process is repeated multiple times. Stage 2 is the core of a transformer model, it *transforms* our inputs into something else.
{{< /timelineItem >}}

{{< timelineItem icon="code" header="Stage 3" badge="Output" subheader="We convert the matrix output from Stage 2 into task-specific results." >}}
  This is usually done by another set of simple matrix operations, depending on the task. For example, if we are doing text sentimental analysis task, this operation could be a simple matrix multiplication, resulting in a final score of 0-10.
{{< /timelineItem >}}

{{< /timeline >}}

In other words, you can concepturally see **Stage 1** as a conversion stage in order to initiate the model, **Stage 2** being the core of a transformer model, and **Stage 3** as a 'decoding' step to convert the output back into human-readable form. When we talk about transformer models, we are mostly referring to  **Stage 2**, which is the focus of the current post. I'll elabrate a lot more on what's hapenning in **Stage 3** in the next post.<br>

### The Transformer Itself
The transformer itself is pretty straightforward: it consists of stacks of multiple attention layers that often share exactly the same, or very similar structure:
{{< mermaid >}}
---
config:
  layout: dagre
---
flowchart LR

    n1["Input"] --> n2["Attention<br>Layer"]
    n2 --> n3["Attention<br>Layer"]
    n3 --> n6["Repeat"]
    n6 --> n5["Output"]

    n2@{ shape: rounded}
    n3@{ shape: rounded}
    n6@{ shape: text}
    classDef default fill: transparent, bg-color: transparent

{{< /mermaid >}}

Asttention layer consists of multiple 'attention heads' that works in parallel: each attention head processes the inputs independently, and the output of all attention heads are merged back together. The joined matrix is the final output of the current layer: <br>

{{< mermaid >}}
flowchart LR
 subgraph s1["Attention Heads"]

    direction LR
        n14["Attention Head"]
        n16["Attention Head"]
        n17["Attention Head"]

  end

    s1 --> n10["Merge"]
    n10 --> n18["Next Layer"]
    n19["Previous Layer"] --> s1

    n18@{ shape: text}
    n19@{ shape: text}
    classDef default fill: transparent
    style s1 fill:transparent

{{< /mermaid >}}

### The attention head
You can think each of the attention head as a mini neural network. A typical attention head works like this:

{{< mermaid >}}
flowchart TB

    IN["Previous Layer"] --> Q["Matrix 1"] & K["Matrix 2"] & V["Matrix 3"]
    Q ---> SDPA["Matrix 1 & 2"]
    K ---> SDPA
    SDPA --> n1["Output"]
    V ---> n1
    classDef default fill: transparent, bg-color: transparent

{{< /mermaid >}}
<br>
1. <bullet> Step 1: The input matrix gets converted into multiple matices through matrix multiplication. Most current transformers converts input matrix into three smaller matrices.</bullet>
2. <bullet>Step 2: Two of the matrix from step (1) gets combined together using some matrix operation, usually dot products.</bullet>
3. <bullet>Step 3: The third matrix from step (1) combines with output from step (2), using some other matrix operation.</bullet>

## Example: Qwen3
We'll now take a look at an actual transformer model and see how it works in action.
It's very suprising is that, transformers are able to produce pretty impressive results for tasks model that are not specifically trained for. Here, we'll use [Qwen3, a small-sized text generation model](https://qwen.ai/blog?id=qwen3) as a demo. It's tiny (~1.5GB) but the performance is VERY impressive for its size.<br>
Here's huggingface's link to the model: <br>
{{< huggingface model="Qwen/Qwen3-0.6B" >}}

### Prep
Make sure you have python pre-installed. If you were using windows, python can be downloaded from the Microsoft Store. If you were Mac/ Linux Mint/ Ubuntu/ Debian etc., your computer should already come with python by default. For this demo, we would need python version <a style="font-size:75%">\( \geqslant \)</a> 3.12.<br>

We first need to install some dependencies. Open terminal/ command prompt, type this command to install required dependencies:<br>

```sh
pip install torch transformers
```
And then type:<br>
```sh
python
```
To open python and start an interactive python session.<br><br>
*Alternatively, if you installed [ipython](https://ipython.org/install.html), which should already be installed if you have installed [jupyter notebook](https://jupyter.org/install) previously, you can open ipython instead:*
```sh
ipython
```

Once python was opened, copy and paste these lines into python to import required packages:
```python
from pprint import pprint
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding
```
<div class="aSeparator"></div>

### Get the model
Models from ```transformers``` library takes matrices as inputs, and outputs matrices. Thus, text generation models needs to be paired with ```tokenizer``` in order to convert inputs into model-readable formant, and convert model outputs into human-readble format. ```transformers``` library has a special class called ```TextGenerationPipeline``` to handle this conversion class, but would be a bit too complicated for our purpose. Here, I modified demo code from <a href='https://huggingface.co/Qwen/Qwen3-0.6B'>Qwen3's demo code</a> that simply combines tokenizers and models together. Here's the code for you to copy and paste into the currently running python session:

```python
class DemoChatbot:
    """
    A simple demo chatbot, code modified from https://huggingface.co/Qwen/Qwen3-0.6B
    """
    def __init__(self, model_name:str="Qwen/Qwen3-0.6B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.enable_thinking = False
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            )
        self.history = []

    def clear_history(self) -> None:
        self.history = []

    def tokenise(self, user_input:str, enable_thinking=None) -> BatchEncoding:
        if enable_thinking is None:
          enable_thinking = self.enable_thinking  # default value

        messages = self.history + [{"role": "user", "content": user_input}]
        tokens = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            enable_thinking=enable_thinking, 
            add_generation_prompt=True,
            return_tensors="pt"
        )
        return tokens

    def __call__(self, user_input, enable_thinking=None) -> None:
        inputs = self.tokenise(user_input, enable_thinking=enable_thinking)
        response_ids = self.model.generate(
            **inputs,
            max_new_tokens=2048,
            )
        response_ids = response_ids[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # since we aren't going to do anything with the output, 
        # just prints out the response and save it to the chat history.
        print(response)

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})
        
```

Then, create a new chat instance with the code below:
```python
chat = DemoChatbot()
```
You will see a progress bar showing download status. Once completed, type the following to have a look at the QWen3 model structure:
```python
pprint(chat.model)
```

…which in term will give you this output:

```python
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

### Model Architecture
The very core of a transformer model can be checked by:
```python
pprint(chat.model.model)
```

And you will get an output similar to the previous output, but without the ```embed_tokens``` at the beginning and the ```lm_head``` at the end:<br>

```python
Qwen3Model(
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
```

The ```embed_tokens``` is the [**Stage 1**](#overview) mentioned earlier, and the ```lm_head``` is the [**Stage 3**](#overview). The core of a transformer, is what we've been discussing today.<br>

Let's have a look at one of the layers inside the transformer:<br>
```python
layer = chat.model.model.layers[0]
pprint(layer)
```

You will see something like this:
```python
Qwen3DecoderLayer(
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
```

Layer is divided into 2 major sections, Just like like we [discussed previously](#the-transformer-itself):
- ```self_attn``` is where attention gets calculatd.
- All the rest (i.e., ```mlp```, ```input_layernorm``` and ```post_attention_layernorm```) are step-by-step computations of combining and averaging the attention heads.

Run the following code to see the structure of an attention head:
```python
attention = layer.self_attn
pprint(attention)
```

The ```q_proj```, ```k_proj``` and ```v_proj``` are the [three matrices](#the-attention-head) mentioned earlier:
```python
Qwen3Attention(
  (q_proj): Linear(in_features=1024, out_features=2048, bias=False)
  (k_proj): Linear(in_features=1024, out_features=1024, bias=False)
  (v_proj): Linear(in_features=1024, out_features=1024, bias=False)
  (o_proj): Linear(in_features=2048, out_features=1024, bias=False)
  (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
  (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
)
```

The actual impelmentations of attention heads are different from model to model, but the general principle behind should be roughly the same. I'll dive depper into it in the future. 

### Have some fun!
Meanwhile, since we've got a generative model already, we might as well test out some text generations:

```python
chat("Why is the content, which was held to be true in perceiving, in fact only belongs to the form, and it dissolves into the form’s unity 🥺🥺🥺🥺🥺??")
```

Or, you can turn on the 'thinking' mode to enable chain-of-thought:

```python
chat.enable_thinking = True
chat("But...but the phenomenology Φ147 says the inner, essential is essentially the truth of appearance😠😠😠 I'm absolutely fewming")
```

If it takes too long to run, you can clear chat history to remove cached chats:
```python
chat.clear_history()
```

![llm is magic text](./featured.jpg "'LLM is magic'")

I wanted to point out the (maybe) obvious thing here: almost all operations mentioned contain learnable parameters. Inside individual attention heads, the three matrices convreted by inputs are typically converted by multiplying ('dot product') inputs with three **separate** matrices. These matrices are part of the learnable parameters for the attention head. When we combine matrices, the combination operation also has [their own learnable parameters](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html). Furthermore, when we combining outputs from each 'attention heads', this combining opearation also has its own set of trainable parameters, so on and so on... Almost every stage of the matrix computations are parameterised, resulting the unbelievably **massive** Aİ models as of today. However, the model used in the demo today, despite it's size, is still able to produce long and very coherent responses. Yes, in a sense, transformers are just a huge stack of matirx calculation? Howver, we know very little about the reason behind We gave names to these matrices, we don't know much about their behaviour.<br>

## Final thoughts

Transformer neural network models aren't as mysterious as one think it would be. It has no difference compared with any other functions. At the end of the day it's just another mathematical function, but takes *matrix* as inputs, does *matrix* calculations, and outputs  *matrices*. Just like any other neural network models, it takes multiple steps to do the calculation. Within each computational step, the inputs gets processed by three mini-neural networks, and outputs are re-combined together as the output of the current step. In this sense, transformers are like nested neural networks: they are bigger neural network that contains lots of mini neural networks.<br>

**What's up next?**<br>
There are still a bit more stuffs that I'd like to share, like how text-generation works and what's really happenning when we are training a generative model. We've heard of the same old things over and over: *'generative LLM is just a very massive auto-complete!'*. Whilst I do aggree with it, I also find it not helpful if one wants to *understand* text generation, for both model training and model inference. In the next post, we'll have a look at the **Stage 3** for text generation.

## Some Other Resources

I hope whoever come accross this post would find it useful. Here are some extra reading materials that İ found particularly useful:<br>

1. [Pytorch's step-byp-step guide](https://docs.pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html) on creating a generative LLM is prob one of the best out there that teaches you all the fundenmentals.
2. [BertViz, a very good visualisation tool](https://github.com/jessevig/bertviz) for looking at attention heads layer by layer. You can run it interactively in a jupyter notebook.
3. [Huggingface's LLM cources.](https://huggingface.co/learn/llm-course/en/chapter0/1) Although they tends to focus on the side of programming & practical applications, I found many of their conceptual guides very good for a beginner.
