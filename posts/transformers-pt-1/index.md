---
title: "ELI5 Transformers (Part 1): Attention Mechanism "
date: 2025-10-31
publishDate: 2025-10-31
draft: false
summary: "(it's glorified linear algebra)"
font_family: "Monospace"
tags: ["huggingface", "AI", "ELI5"]
category: ["ELI5-Transformers",]
build:
  list: always
params:
  math: true
---
As someone without much backrounds in neither physics nor computer science, I find lots of available introductions on transformers very confusing. Transformers is the talk of the street, GPT is short for 'Generative Pre-training <i>Transformer</i>'! However, most of the articles on transformers focuses on attention mechanisms, using either [the OG transformer](https://peterbloem.nl/blog/transformers) or [the classic BERT](https://towardsdatascience.com/a-complete-guide-to-bert-with-code-9f87602e4a11/) as examples. They would spend lot of time talking about embeddings & attetions on the encoding side, and skipped most the decoding by saying 'well you just do the same thing again and there you have it!'. Well that's not very helpful isn't it. Don't get me wrong, there are a lot of very good learning materials out there, for example the amazing [interactive transofmer explainer](https://poloclub.github.io/transformer-explainer). However, I always find these heavy tutorials not very suitable for my very short attention span or the autism tendency of getting lost in details.<br>

Finally, I've decided to bite the bullet and spend some time have a read through the HuggingFace's source code for google's T5 Model, [THE encoder-decoder everyone on the steet are talking about](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py). It was a relatively long process with lots of back and forth jumping between classes and methods. I could not emphasis how much I appretiate HuggingFace's [maximalist coding choice](https://huggingface.co/blog/transformers-design-philosophy), where the entire model architecture is contained inside one single .py file. The bonouns point is, I didn't experience the pain of come accross `import tensorflow` [inside dataset iteration](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/data/data_collator.py#L742). <br>

I took lots of notes here and there during the process of studying transformers, think now it's a very good time to share some of my findings. In this (or probably a series of?) blogpost(s?), I am going to collate my past notes on text-specific transformer models piece by piece in a reader-friendly manner. I hope these notes can help others alongside their studying, or being an interesting nice little piece of articles to read through.<br>

In this particular post, I would like to do a very brief overview of the transformer model architecture, specifically on the attention mechanism. I won't go metion too much math and there won't be any mathenathical formulas. However, I would assume readers of this silly little post already have some okay-ish background of math/ datascience. (i.e., matrix computations embeddings, tokens, model fitting etc.). I am not going to list out all the implemention details for transformers, since there are a lot of very good materials out there and they are doing fantastic jobs. Instead, in this (maybe series of?) post, İ would like to draw out a general framework on transformers to help one understand the detailed math behind.<br>

In this particular blogpost, I'll very quickly go through some very basics on neural network model, just enough to cover what needed for this post, accompied by demo of transformer model as a proof of concept. In this section, there will be some codes that you can copy and paste into an interactive python session to fiddle around for a bit. And lastly, I'll do a quick sketch on the general architecture of transformers, and a overview of the attention mechasm.<br>

## A good model only needs to be useful
In order to make things easier to understand, I would wish to start with an inaccuate premise: we can view neural network models as functions that takes some sort of matrix as inputs, do some sort of matrix computations, and output another matrix as the final result. What makes one neural network different from others is how the computation is carried out. It's like \(y=a \times x^2\) is a different function from \(y=a \times sin(x)\), only that in the case of neural network, both x and y are matrics, and the math is much complicated. When it comes to model training, we are essentially trying to find values gives best fit to the data.<br>

There's an important assumption here: just because model fits the data well does not mean the model describes mechanisms behind the data. For example, we <i>definitely</i> can fit \(y=a \times sin(x) + b\) to a normal distribution data (like distribution of customer spendings in McDonald's), and it's prob going to be a pretty good fit, but this does not mean the sine function has anything to do with explaining the normal distribution. A good model does not always need to be description, a good model just needs to be useful for its purpose.<br>

Transformers are preciesly these kinds of models: they are, surprisingly good at fitting into all sorts of data whilst the math behind the model probably doesn't have much to do with the mechanisms behind. We don't know how transformers works so well for text-based tasks. At least not yet. Originally, transformer was designed as an add-on to the text-processing neural network models in order to tackle with some tricky problems (these problems are not the main forcus of the current blogpost so I'm skipping them, but [here's a good article if you were interested](https://towardsdatascience.com/beautifully-illustrated-nlp-models-from-rnn-to-transformer-80d69faf2109/)). We just happened to discover that transformers alone is good enough to solve these problems, we just need to make the transformers much bigger. So that's where the Aİ bloom started: GPT2 solved issues in GPT1 by simply being 10 times bigger; the most-recently open-sourced [pretrained GPT-Oss](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1), is 200 times bigger than [the previous openpsourced model, GPT2](https://huggingface.co/openai-community/gpt2) <i>(note: GPT-OSS is structurlly different from the original GPT2 but the fundamental ideas are the same.)</i>. There are even speculations suggesting transformer neural network models can be seen as some sort of universal function approximator. That is, it's capacable of 'approximate' other formulas/ functions with certain degree of accuracy, providing the model itself is big enough (['universal approximation theorem'](https://en.wikipedia.org/wiki/Universal_approximation_theorem)). <br>

#### Demo: Machine translation with T5
It is preciesly the reason why it's very suprising is that, transformers are able to produce pretty impressive results for tasks model that are not specifically trained for. You can try this out yourself. I'll use Flan-T5 as a demo here. Flan-T5 is a variation of T5 model that fine-tuned on instruction-specific tasks. That is, we can insert some instructions before our prompt and the model shall return different results based on different instructions.<br>

I'll use Python for the demo here because it's convinent. Before starting, you might want to install `transformers` if you haven't done aleady. It's a library collecting huge tone of open-sourced transformer models that allows you explore around.<br>
Open terminal, run this command to install transformers:
```sh
pip install torch
pip install transformers
```
The model I am going to use is [T5, released by Google a couple of years ago](https://arxiv.org/pdf/2210.11416) Here's huggingface's link to the model:
{{< huggingface model="google/flan-t5-base">}}

In python, run these lines to download & initialise the model:
```python
from transformers import pipeline
import pprint  # to print indented dictionary
pipe = pipeline('text2text-generation', model="google/flan-t5-base")
```
T5 is one of the very few models that comes with very well-documented records on what kind of task the particular model has been trained on. To view the list of tasks the original T5 model fine-tuned on, we can check the 'task_specific_params' attribute in model.config:
```python
pprint.pp(pipe.model.config.task_specific_params)
```
Output:
```python
{'summarization': {'early_stopping': True,
                   'length_penalty': 2.0,
                   'max_length': 200,
                   'min_length': 30,
                   'no_repeat_ngram_size': 3,
                   'num_beams': 4,
                   'prefix': 'summarize: '},
 'translation_en_to_de': {'early_stopping': True,
                          'max_length': 300,
                          'num_beams': 4,
                          'prefix': 'translate English to German: '},
 'translation_en_to_fr': {'early_stopping': True,
                          'max_length': 300,
                          'num_beams': 4,
                          'prefix': 'translate English to French: '},
 'translation_en_to_ro': {'early_stopping': True,
                          'max_length': 300,
                          'num_beams': 4,
                          'prefix': 'translate English to Romanian: '}}
```
These configurations are task-specific parameters for [text-generation](https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationMixin.generate). Each item in the dictionary <i>('translation_en_to_de', 'summarization' etc., )</i> corresponds to each text-generation task that the model been previously trained on. As shown in the code block above, the particular model we are testing today was trained on summarization, and three translation tasks: English-German, English-French, and English-Romanian.What we are interested is the 'prefix' key under task name <i style="0.8em">(i.e., 'translate English to Romanian: ')</i>. These are the texts that inserted at the beginning of every text input, as extra instructions of telling our model a bit more information about what it should do.<br>
For example, if one wants to do english-to-romanian translation, the model input would be converted as follows:<br>
        'I LOVE FISH!!!!' --> <i>'translate English to German: I LOVE FISH!!!!'</i><br>


..So instead of languages model already trained on, let's try something model never trined before: English to Spanish translation:
```python
print(pipe('translate English to Spanish: I love fish!!!!!'))
```

Althogh the model was not trained on Spanish translation tasks, it still produced pretty impressive results:
```json
[{'generated_text': 'Me encanta el pescado!'}]
```

It's very uncanny that model is able to do things us human did not ask it to do. There are lot of speculations on why model is able to perform such tasks. For exmaple, some researchers do suggest models that are big enough might [capture meanings behind words as well as language-specific syntax features](https://aclanthology.org/W19-4828/), and thus are able to convert one language to another. You can view how big the model in the demo is:

```python
pprint.pp(f"Number of parameters: {pipe.model.num_parameters():,}")
```
Output:
```python
'Number of parameters: 247,577,856'
```

![llm is magic text](./featured.jpg "'LLM is magic'")

<!--  -->
## The Transformer Architecture
### Overview: *the* transformer itself
We'll now take a look inside the transformer models and see what kind of math calculations is hapenning. `pytorch`, the python package that the T5 model in this demo is based off, provies very good tool for visulising model structures Using models mentioned from the previous section, if you want to look at what the model actually lookes like, you can do so by running:

```python
pprint.pp(pipe.model)
```
...which in term will give you this monstrous output:

```python
T5ForConditionalGeneration(
  (shared): Embedding(32128, 768)
  (encoder): T5Stack(
    (embed_tokens): Embedding(32128, 768)
    (block): ModuleList(
      (0): T5Block(
        (layer): ModuleList(
          (0): T5LayerSelfAttention(
            (SelfAttention): T5Attention(
              (q): Linear(in_features=768, out_features=768, bias=False)
              (k): Linear(in_features=768, out_features=768, bias=False)
              (v): Linear(in_features=768, out_features=768, bias=False)
              (o): Linear(in_features=768, out_features=768, bias=False)
              (relative_attention_bias): Embedding(32, 12)
            )
            (layer_norm): T5LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): T5LayerFF(
            (DenseReluDense): T5DenseGatedActDense(
              (wi_0): Linear(in_features=768, out_features=2048, bias=False)
              (wi_1): Linear(in_features=768, out_features=2048, bias=False)
              (wo): Linear(in_features=2048, out_features=768, bias=False)
              (dropout): Dropout(p=0.1, inplace=False)
              (act): NewGELUActivation()
            )
            (layer_norm): T5LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
      (1-11): 11 x T5Block(
        (layer): ModuleList(
          (0): T5LayerSelfAttention(
            (SelfAttention): T5Attention(
              (q): Linear(in_features=768, out_features=768, bias=False)
              (k): Linear(in_features=768, out_features=768, bias=False)
              (v): Linear(in_features=768, out_features=768, bias=False)
              (o): Linear(in_features=768, out_features=768, bias=False)
            )
            (layer_norm): T5LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): T5LayerFF(
            (DenseReluDense): T5DenseGatedActDense(
              (wi_0): Linear(in_features=768, out_features=2048, bias=False)
              (wi_1): Linear(in_features=768, out_features=2048, bias=False)
              (wo): Linear(in_features=2048, out_features=768, bias=False)
              (dropout): Dropout(p=0.1, inplace=False)
              (act): NewGELUActivation()
            )
            (layer_norm): T5LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (final_layer_norm): T5LayerNorm()
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (decoder): T5Stack(
    (embed_tokens): Embedding(32128, 768)
    (block): ModuleList(
      (0): T5Block(
        (layer): ModuleList(
          (0): T5LayerSelfAttention(
            (SelfAttention): T5Attention(
              (q): Linear(in_features=768, out_features=768, bias=False)
              (k): Linear(in_features=768, out_features=768, bias=False)
              (v): Linear(in_features=768, out_features=768, bias=False)
              (o): Linear(in_features=768, out_features=768, bias=False)
              (relative_attention_bias): Embedding(32, 12)
            )
            (layer_norm): T5LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): T5LayerCrossAttention(
            (EncDecAttention): T5Attention(
              (q): Linear(in_features=768, out_features=768, bias=False)
              (k): Linear(in_features=768, out_features=768, bias=False)
              (v): Linear(in_features=768, out_features=768, bias=False)
              (o): Linear(in_features=768, out_features=768, bias=False)
            )
            (layer_norm): T5LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (2): T5LayerFF(
            (DenseReluDense): T5DenseGatedActDense(
              (wi_0): Linear(in_features=768, out_features=2048, bias=False)
              (wi_1): Linear(in_features=768, out_features=2048, bias=False)
              (wo): Linear(in_features=2048, out_features=768, bias=False)
              (dropout): Dropout(p=0.1, inplace=False)
              (act): NewGELUActivation()
            )
            (layer_norm): T5LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
      (1-11): 11 x T5Block(
        (layer): ModuleList(
          (0): T5LayerSelfAttention(
            (SelfAttention): T5Attention(
              (q): Linear(in_features=768, out_features=768, bias=False)
              (k): Linear(in_features=768, out_features=768, bias=False)
              (v): Linear(in_features=768, out_features=768, bias=False)
              (o): Linear(in_features=768, out_features=768, bias=False)
            )
            (layer_norm): T5LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): T5LayerCrossAttention(
            (EncDecAttention): T5Attention(
              (q): Linear(in_features=768, out_features=768, bias=False)
              (k): Linear(in_features=768, out_features=768, bias=False)
              (v): Linear(in_features=768, out_features=768, bias=False)
              (o): Linear(in_features=768, out_features=768, bias=False)
            )
            (layer_norm): T5LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (2): T5LayerFF(
            (DenseReluDense): T5DenseGatedActDense(
              (wi_0): Linear(in_features=768, out_features=2048, bias=False)
              (wi_1): Linear(in_features=768, out_features=2048, bias=False)
              (wo): Linear(in_features=2048, out_features=768, bias=False)
              (dropout): Dropout(p=0.1, inplace=False)
              (act): NewGELUActivation()
            )
            (layer_norm): T5LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (final_layer_norm): T5LayerNorm()
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (lm_head): Linear(in_features=768, out_features=32128, bias=False)
)
```
Despite its indimidate looks, at conceptual level, our T5 model is actually pretty simple. It does have a lot of layers, however, most of the layers are more or less the same, apart from the very beginning (```Embedding(32128, 768)```) and the very end (```lm_head```). Not only just T5, most of the popular transformer models that you heard of (GPT, Gemmini etc.,) all share similar three-stage structure like T5. Here's a simplified flow chart:

<body>
    <pre style="font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><code>╭─────────────────────────────────────────╮
│                Embedding                │
╰─────────────────────────────────────────╯
╭─────────────────────────────────────────╮
│                    ↓                    │
╰─────────────────────────────────────────╯
╭──────────── Attention Layer ────────────╮
│ ╭───────────╮╭───────────╮╭───────────╮ │
│ │ Attention ││ Attention ││           │ │
│ │ Head      ││ Head      ││ ...       │ │
│ ╰───────────╯╰───────────╯╰───────────╯ │
╰─────────────────────────────────────────╯
╭─────────────────────────────────────────╮
│                    ↓                    │
╰─────────────────────────────────────────╯
╭──────────── Attention Layer ────────────╮
│ ╭───────────╮╭───────────╮╭───────────╮ │
│ │ Attention ││ Attention ││           │ │
│ │ Head      ││ Head      ││ ...       │ │
│ ╰───────────╯╰───────────╯╰───────────╯ │
╰─────────────────────────────────────────╯
╭─────────────────────────────────────────╮
│                    ↓                    │
╰─────────────────────────────────────────╯
╭─────────────────────────────────────────╮
│                   ...                   │
╰─────────────────────────────────────────╯
╭─────────────────────────────────────────╮
│                    ↓                    │
╰─────────────────────────────────────────╯
╭─────────────────────────────────────────╮
│                 Output                  │
╰─────────────────────────────────────────╯
</code></pre>
</body>

These three stages are:

1. <bullet>*The input gets converted into vectors or matrics.*<br>
  The first step starts with creating a mathematical representation of our input data. This process can vary based on different types of inputs. It can simply be some sort of look-up tables (text embedding), some matrix transformations of the raw inputs (convolution) etc. <br>
  In our T5 mdoel, this corresponds to the *```Embedding(32128, 768)```* shown in our T5 model, or the *'Embedding'* in the flow chart.<br></bullet>
2. <bullet>*The raw output from Stage 1 feeds into the multiple different attention layers.*<br>
  Mathematically, each attention layer is doing very much the same mathematical operation, with each layer having its own sets of parameters. Each layer takes a matrix as an input, and outputs another matrix to pass onto the next layer. This process is repeated multiple times. Stage 2 is the core of a transformer model, it *transforms* our inputs into something else.<br>
  In our T5 model, this coressponds to the *```encoder```* and the *```deocoder```*, or the *'Attention Layers'* in the flow chart. 
  <bullet>
3. <bullet>*We convert the matrix output from Stage 2 into task-specific results.*<br>
  This is usually done by another set of simple matrix operations, depending on the task. For example, if we are doing text sentimental analysis task, this operation could be a simple matrix multiplication, resulting in a final score of 0-10.<br>
  In our T5 model, this coresspinds to the *```lm_head```* at the end of the T5, or the *Output* in the flow chart.
  </bullet><br>

In other words, you can concepturally see **Stage 1** as a conversion stage in order to initiate the model, **Stage 2** being the core of a transformer model, and **Stage 3** as a 'decoding' step to convert the output back into human-readable form. The rest of the post will focus on **Stage 2**, the core, and I'll elabrate a lot more on what's hapenning in **Stage 3** in the next post.

### The attention head
Each attention layer consists of multiple attention heads *(```T5Attention```)*, each of them operated independent from one another: they take output matrix from the previous layer as the input, do some mathematical calculations, and output the transformed matrix. Outputs from each of the attention heads are then joined together *(```layer_norm```)*. The joined matrix is the final output of the current layer. <br>

You can think each of the attention head as a mini neural network. It takes some inputs and spits out some outputs.<br>

A typical attention head works like this:

1. <bullet>The input matrix gets converted into multiple matices through matrix multiplication. Most current transformers converts input matrix into three smaller matrices.</bullet>
2. <bullet>Two of the matrix from step (1) gets combined together using some matrix operation, usually dot products.</bullet>
3. <bullet>The third matrix from step (1) combines with output from step (2), using some other matrix operation.</bullet>
<body>
    <pre style="font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><code style="font-family:inherit">╭──────────────────────────────────────────────╮
│              * previous layer *              │
╰──────────────────────────────────────────────╯
     ↓           ↓                              
╭──────────╮╭──────────╮                        
│ Matrix 1 ││ Matrix 2 │           ↓            
╰──────────╯╰──────────╯                        
     ↓           ↓                              
╭──────────────────────╮╭──────────────────────╮
│ Matrix 1 &amp; 2 Merged  ││       Matrix 3       │
╰──────────────────────╯╰──────────────────────╯
           ↓                       ↓            
                                                
╭──────────────────────────────────────────────╮
│          Matrix 1 &amp; 2 &amp; 3 Re-Joined          │
╰──────────────────────────────────────────────╯
╭──────────────────────────────────────────────╮
│                    ↓ ↓ ↓                     │
╰──────────────────────────────────────────────╯
╭──────────────────────────────────────────────╮
│              * to be combined *              │
╰──────────────────────────────────────────────╯</code></pre>
</body>

## Why are the models so big?
I wanted to point out the (maybe) obvious thing here: almost all operations mentioned contain learnable parameters. Inside individual attention heads, the three matrics convreted by inputs are typically converted by multiplying ('dot product') inputs with three **separate** matrices. These matrics are part of the learnable parameters for the attention head. When we combine matrices, the combination operation also has [their own learnable parameters](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html). Furthermore, when we combining outputs from each 'attention heads', this combining opearation also has its own set of trainable parameters, so on and so on...Almost every stage of the matrix computations are parameterised, resulting the unbelievably **massive** Aİ models as of today.

**So, it's just a huge stack of matirx calculation?**<br>
In sense yes, it is. The three matrics in the attention head are commonly named as 'key'. 'query' and 'value', the idea behind these names are: 'user queries something, program looks for keys (i.e., keywords) to match with query, and value is whatever gets matched with'. Honstly I found this explaination very confusing, although by design, it does (sort of) work in such way. My main skeptcisim is that, just because we vaguely designed it this way does not mean it is really what's happening underneath. As shown in the demo earlier, a model that's not trained for English-Spanish translation did have some ability to do English-Spanish translation, innit. We gave names to these matrics, we don't know much about their behaviour.<br>

## What's up next?<br>
I think this is a rather good place to conclude this post, so I'll leave it here for now. I hope you find it helpfull.<br>

There are still a bit more stuffs that I'd like to share, like how text-generation works and what's really happenning when we are training a generative model. We've heard of the same old things over and over: *'generative LLM is just a very massive auto-complete!'*. Whilst I do aggree with it, I also find it not helpful if one wants to *understand* text generation, for both model training and model inference. In the next post, we'll have a look at the **Stage 3** for text generation.

## Some Other Resources
I hope whoever come accross this post would find it useful. Here are some extra reading materials that İ found particularly useful:<br>
1. [Pytorch's step-byp-step guide on creating a generative LLM is prob one of the best out there that teaches you all the fundenmentals.](https://docs.pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html)
2. [BertViz, a very good visualisation tool for looking at attention heads layer by layer. You can run it interactively in a jupyter notebook.](https://github.com/jessevig/bertviz)
3. [Huggingface's LLM cources. Although they tends to focus on the programming & practical applications, I found many of their conceptual guides very good for a beginner.](https://huggingface.co/learn/llm-course/en/chapter0/1)

