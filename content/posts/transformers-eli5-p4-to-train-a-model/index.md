---
title: "ELI5 Transformers - to Train a Model"
date: 2026-02-04
publishDate: 2026-02-04

summary: "Overview of all the ins and outs for training AI models."
tags: ["huggingface", "ELI5",]
topics: "transformers"
draft: false

featureimage: "https://imgs.xkcd.com/comics/aurora_coolness.png"
---

{{< katex >}}

![cover](https://imgs.xkcd.com/comics/aurora_coolness.png "[relevant xkcd](https://xkcd.com/3196/)")

Training is usually the thing that most data scientists forgets about, since majorities of the packages handles the training for us. One would at most provide some estimations of the parameters/ parameter distributions and let the package do the rest: Stan's ```stan()```, Jags' ```jags(jags.data)```, scikit's ```curve_fit```, or Keras' ```Model.fit```. Trainning has always been this *'black box'* where people forget about. However, this shouldn't be the case when it comes to transformers. However, when it comes to transformers, training is probably as important as the data used for training. It's the black box that one *absolutely* needs to get into, in order to be able to create a good, functional transformer model.


This post will give a very gentle introductions by covering the very basics of model training. Specifically, this post will

- Introduce the basic idea of transformers as universal function approximator, and explain why we don't care as much about the model architecture as much.
- Overview on how transformer models are evaluated, and what typical model training workflow looks like.
- Going into an individual optimisation step, look at how prameters are updated based on model performance.
- Provides a conceptual gudide on training hyperparameters: what are they, what are they used for etc.,

> [!note] 
> *Although code examples in this post are based off ```pytorch``` library, these examples are mainly used for the purpose of providing better clearification on complex concepts, instead of practical guidelines. You do not need to have any familarity of ```pytorch``` library to follow through this post.*

## Background: Universal Approximation

### The blackbox

Before looking into how training works, let's first start with a thought experiment. Let's say, you want to a model that converts **digits** like [1, 2, 3,..] into **arabic numerals**  [٣ , ٢ , ١, ...]:

<style type="text/css">
.tg th{border-color:black;border-style:line;border-width:1px;
  padding:10px 25px;word-break:normal;}
.tg .tg-c3ow{border-color:inherit;text-align:left;vertical-align:top}
</style>
<table class="tg"><thead>
  <tr>
    <th class="tg-c3ow">Inputs</th>
    <th class="tg-c3ow">0</th>
    <th class="tg-c3ow">1</th>
    <th class="tg-c3ow">2</th>
    <th class="tg-c3ow">...</th>
  </tr></thead>
<thead>
  <tr>
    <th class="tg-c3ow">Desired outputs</th>
    <th class="tg-c3ow">٠</th>
    <th class="tg-c3ow">١</th>
    <th class="tg-c3ow">٢</th>
    <th class="tg-c3ow">...</th>
  </tr>
</thead>
</table>

This model could be as simple as an **if-else** check or a lookup-table that gives you specific output when seeing specific input:
{{< mermaid >}}
flowchart LR
    ipt(["Input"])
    ipt --> if1["Is input 0?"]
    if1 --> o1["Output: ٠"]
    ipt --> if2["Is input 1?"]
    if2 --> o2["Output: ١"]
    ipt --> if3["..."]
    if3 --> o3["..."]
    classDef default fill: transparent, bg-color: transparent
{{< /mermaid >}}

Or, this could be a very complicated LLM that does lots of unnecessary thinking:

```fish
First, the user said: "Convert digits to arabic numerals:" and then listed "- 0", "- 1", "- 2". I need to understand what is being asked here. The term "arabic numerals" might be confusing because in common usage, the digits 0, 1, 2 are already Arabic numerals. But perhaps the user is referring to the numerals used in Arabic script, which are different from the Western Arabic numerals we use.

In Arabic, numerals are written differently. For example, the Western Arabic numeral "0" is represented as "٠" in Arabic script, "1" as "١", "2" as "٢", and so on. So, probably, the user wants the conversion from Western digits to Eastern Arabic numerals.

Let me confirm: In many Arabic-speaking countries, they use these numerals: ٠, ١, ٢, ٣, ٤, ٥, ٦, ٧, ٨, ٩. So for the given digits:
  0 should be converted to ٠
  1 should be converted to ١
  2 should be converted to ٢
The user only provided three digits: 0, 1, and 2. So I should only convert these.
The instruction is "Convert digits to arabic numerals", and since it's listed with hyphens, I should probably respond in a similar format.
So, for each digit, I'll provide the corresponding Arabic numeral.
My response should be clear and direct.

The digits 0, 1, and 2 in Western Arabic numerals correspond to the following Eastern Arabic numerals used in Arabic script:
    0 → ٠
    1 → ١
    2 → ٢
```

In real life, we often don't have the formula for our target function \(F\), but, if we were only interested in getting an acceptable \(y\), then an approximation of \(F\) would be decent enough. Think about the digital-arabic converter in our example: both lookup table and the state-of-art AI give us satisfying answer, they provided the same output with the same input. All LLMs, from big names like GPT or Gemini, to domain-specific models like [Helium](https://huggingface.co/docs/transformers/model_doc/helium), shares this one key aspect in common: they are set up to produce **specfic outputs** with **specific inputs**. Most of the time we aren't interested in how these outputs are produced, we only need to the output to be accurate.

Here, 'most of the time' applies for **both** model inference (when running the model) and model **training**. There is this idea of [universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem). Suppose we have a target function, say, \(y = F(x)\), and we have our transformer \(y = T(x)\). With sufficient training and sufficient model size, taking the same input \(x\), our transformer \(T\) would be able to produce similar output \(y\) as the target function \(F\). Most of the time we don't really need to think too much about model architectures.

Since we are **only** interested in function inputs and function outputs, transformers comes very handly for creating AI models that suits our specific to our need: we only need to set up appropriate training data, feed the data to the model, and we would expect model behave in ways we want without worrying too much about building an adequate model.

> [!note]
> *There are times where model architecture does matter. For example, from my experience, models with bidirectional attention (i.e. BERT Encoder) are much robust for token classification/ named entity tasks compared with models with causal masked models (i.e., decoder-only models like GPT/ QWen/ Deepseek).*<br>
> *Underperformed models usually are capacable of achieving similar outcome, just need to be bigger.*

## Training: overview

### Goodness of fit

Training is essentially finding the best combination of model parameters that fit the data. But before everything else, one would first need to define what the goodness of fit would be. Usually this means checking the output from the model against the target via a fixed set of metrics. For example, this metric can be as simple as checking how far away the model's prediction is from the true datapoint (*mean square error*); the lieklihood of observing *some* variables given *some* proposed parameter distributions (*bayesian*), etc.

Performance metrics for training transformers can vary a lot depending on the end goal. When people talk about 'metrics' this could mean different things for different stages of training a transformer model. This can be generally categorised into two different kinds: the ones that used for *optimising* the model that actually updates the model parameters, and the ones used for *assessing* the performance of model checkpoints on the desired tasks.

1) The **optimisation** kind is not very much different from training any other mathematical models, in addition to [MSE](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html), there are probability based [cross entropy](https://en.wikipedia.org/wiki/Cross-entropy), [Kullback-Leibler divergence](https://docs.pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html).

2) The **performance assessment** kind is highly dependent on the target of the model, there are very frequency-based [f1 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) that used for classification,[BLEU](https://en.wikipedia.org/wiki/BLEU) for translation, etc.

```1``` updates the raw number, whilst ```2``` provides useful key metrics as well as potential problems of the model. The training process usually involves an intermix between above ```1``` and ```2```. One would usually run ```1``` for a certain number of steps, then use ```2``` to assesses model performance, save these parameters as a checkpoint; then switch back to ```1``` and carry on training.

### Batching

Splitting data into minibatches is probably something most non-neural network people wouldn't do a lot. The model is fit with the entire data all at once. The biggest challenge of this with transformers is that dataset can be monumentally gigantic, and it is simply impossible to load this much of data on the computer memory! I did a little bit of test myself, the raw outputs for 16 rows of short texts can be around 20~30MB in size, and my dataset has around 20,000 rows!

So instead of feeding the entire dataset to the model all at once, models are trained with slices data.

### Checkpoints?

The entire training workflow goes something like this:

{{< mermaid >}}
flowchart TB
    B(["batch data"]) e1@==> n1(["model"])
    n1 e2@== Running ==> n2(["output"])
    n2 e3@== optimisation ==> n3(["update parameter"])
    n3 -. after n batches .-> n4(["assessment"]) -.-> n5(["checkpoint"])
    n3 e4@== next bactch ==> B

    n5@{ shape: braces}
    e1@{ animation: fast } 
    e2@{ animation: fast }
    e3@{ animation: fast } 
    e4@{ animation: fast }
    classDef default fill: transparent, bg-color: transparent

{{< /mermaid >}}

Usually models are released with one or more checkpoints that saved following ```2```:
<figure>
<img src=model-checkpoints.png alt="checkpoints" align="middle" style="width:80%">
<figcaption><a href=https://huggingface.co/LSX-UniWue/ModernGBERT_1B/tree/main>example model release</a></figcaption>
</figure>

> These checkpoints can be very useful for estimating the cost of time/ resources for adapting the model for your use case!

## Optimisation step

This section will take a more detailed look at the **optimisation** process, where model's parameter gets updated.

### An optimisation step

It's much easier to understand optimisation process with codes by the side. ```pytorch```'s implementation is the cleanest I've seen. A minimal optimisation loop looks like this:

```python
for data_inputs in all_data:  # run data as mini-batches
    model_outputs = model.forward(data_inputs)
    loss = evaluation_function(model_outputs, data_inputs)
    loss.backward()
    optimizer.step()
    model.zero_grad()
```

Let' go through the codes in side this for-loop line-by-line, and see how parameter gets updated:

***

**model.forward**

```python {linenos=inline hl_lines=1}
model_outputs = model.forward(data_inputs)
loss = evaluation_function(model_outputs, target)
loss.backward()
optimizer.step()
model.zero_grad()
```

```model.forward``` passes ```data_inputs``` to the current model to get predictions with current set of model parameter.

***

**evaluation_function**

```python {linenos=inline hl_lines=2}
model_outputs = model.forward(data_inputs)
loss = evaluation_function(model_outputs, target)
loss.backward()
optimizer.step()
model.zero_grad()
```

*some_evaluation_function* compares model outputs with the correct result, and returning the differences between these two as ```loss```.

***

**loss.backward**

```python {linenos=inline hl_lines=3}
model_outputs = model.forward(data_inputs)
loss = evaluation_function(model_outputs, target)
loss.backward()
optimizer.step()
model.zero_grad()
```

Numbers from the ```pytorch``` library are very different from other libraries, including the ```loss``` variable in our example. In addition to its numerical value, ```loss``` also records the entire trace of how it gets calculated, from ```data_inputs```, through all the parameters used for getting the ```model_outputs```, and ```evaluation_function```.

```loss.backward``` essentially 'announces' its own value to all the parameters involved in the process, providing them a feedback of how well each of them did, are they too big/ too small, and which direction should they move in order to fit the data better.

This boardcasting process is based on some crazily genius math, [which I would urge you give it a watch](https://youtu.be/QwFLA5TrviI), and a detailed explainations of this algorithm implemeentation can be found in [pytorh's excellent guide](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) on the ```autograd``` function.

> [!note]-Example<br>
> Say you have a function, \(y = a \times x + b \). <br> You provide this function an ```input``` and thus get model's ```prediction```. ```loss``` is calculated via:<br> 
> \(loss = target - prediction\)<br>
> Variable ```loss``` contains:
> 1. It's numeric value
> 2. Something that tells python "I am coming from [\(a, b, target\)]"
> 
> Method ```loss.backward()``` essentially tells ```a``` and ```b``` something like "your value too high, get lower", or "your value is too low, move higher",

***

**optimizer.step**

```python {linenos=inline hl_lines=4}
model_outputs = model.forward(data_inputs)
loss = evaluation_function(model_outputs, target)
loss.backward()
optimizer.step()
model.zero_grad()
```

```optimizer``` basically tells the model to update its parameters according to the feedback they received from ```loss.backward```, it also controls how much parameters should update ('learning rate') based on these feedbacks from the previous step.

***

**model.zero_grad**

```python {linenos=inline hl_lines=5}
model_outputs = model.forward(data_inputs)
loss = evaluation_function(model_outputs, target)
loss.backward()
optimizer.step()
model.zero_grad()
```

```model.zero_grad``` clears out the beedback from previous ```loss.backward``` so that the updated paramter can be ready for the next round of evaluation and updates.

***

## The many configurations

The main reason why optimisation steps in ```pytorch``` designed in such way is that, this seemly complicated setting gives one very fine-control over some of the most important key things happenning during model training.

This section will walk through the core importants bits for training process, as well as recommending some of my current approaches.

***

For setting up your own model training, I would strongly recommend you have a read on the paper of your model, and use the original learning rate as a reference for training your model. These settings would usually be great startig points for your own use case.
> Example: [training setting section](https://arxiv.org/pdf/2412.13663#appendix.A) for [ModernBert](https://huggingface.co/answerdotai/ModernBERT-base) model.

***

### Logging

One thing people tends to overlook is the importants of logging. Training is not an one-off process. One need to constantly monitor model's performance and make adequate adjustments accordingly. Just like how lab journalling is the foundational quality of a good bio-chemist, a good habit of logging model training is also the fundation of a good data scientist.

When it comes to training transformers, in general, there should be a file in your model training folder that logs all the key informations about the model, such that you should ALWAYS be able to replicate the trianing from your log. There are some very easy to miss things that I learned by the hard way.

Your log files should have:

- All the arguments and keyword arguments used to initialise your model (```model.__init__```).
- All your data-related configurations, what dataset was used for training, batch size, etc.
- All the hyper parameters used in your training instance. We'll go through some of these hyper parameters later in the same section.
- You also want to keep a record of your model's performance over time. There are a lot of tools for this, like [tensorboard](https://docs.pytorch.org/tutorials/intermediate/tensorboard_tutorial.html), [wandb](https://docs.wandb.ai/models/track), the paid [dagshub](https://dagshub.com/), or [trackio](https://huggingface.co/docs/trackio/index).
- Moreover, DO NOT FORGET TO SAVE A COPY OF **ALL YOUR CLASSIFICATION LABELS**, AND DON'T FORGET TO SAVE THE **TOKENIZER!!** These things holds the key to understand your model output, and if you forget save them somewhere your entire training is WASTED!! I learned it in an extremely hard way, and you definitely do NOT want to be like me 😭.

***

### Training steps

One very good advantage of ```pytorch```'s optimisation syntax is that it allows one to control when model's parameter gets updated. Instead of the example optimisation loop [mentioned before](#an-optimisation-step), one could also do something like this, note the highlighted line 6:

```python {linenos=inline hl_lines=6}
for batch_index, batch_input in enumerate(dataset):
    model_outputs = model.forward(data_inputs)
    loss = evaluation_function(model_outputs, data_inputs)
    loss.backward()

  if batch_index % 5 == 0:
    optimizer.step()
    model.zero_grad()
  else:
    continue
```

Conditional statements in line 6 (```batch_index%5 == 0```) evaluates to be ```True``` for every 5th batch. Meaning model would **only** updates its parameter for every **5** batches of data. This would come very handy when it comes to training very big models, since ```optimizer.step()``` can be very resource-intensive.

***

### Checkpoints

[As discussed earlier](#checkpoints), you would also want to be able to save your model so that you can, of cource, use your model,,, In practice, model are saved at every x checkpoints, so that one could pick up and re-start training from a desired time stamp. Something like:

```python {linenos=inline hl_lines=4}
for batch_index, batch_input in enumerate(dataset):
  ...
  if batch_index % 1000 == 0:  # saves model at every 1000th step
    save_checkpoint(model, path=f"trainning_folder/checkpoint/{batch_index}")
```
... which saves model at every 1000th step.

***

### Evaluation

As mentioned in the [workflow section](#checkpoints) above, one would also need to evaluate model performance every now and then. This is usually handeled like this:

```python {linenos=inline hl_lines=4}
for batch_index, batch_input in enumerate(dataset):
  ...
  if batch_index % 1000 == 0:  # evaluates model at every 1000th step
    performance = evaluate_model(model)
    # Don't forget logging performance merics!
    log_performance(performance, path=f"trainning_folder/performance/{batch_index}")
```

***


### Batch size & Learning rate

![data points](https://imgs.xkcd.com/comics/data_point.png "relevant [xkcd](https://xkcd.com/2713/)")

- Batch size controls how many data are your model going to process at once, before model parameter gets updated via ```optimizer.step```.
- Learning rate, on the other hand, controls *how much* would parameter updates for every ```optimizer.step```. 

Batch size and learning rate should be configured with each other in mind. Larger batch size means model has seen more data, thus 'accumulated' more 'past experience' via ```loss.backward```, therefore would need larger learning rate, vice versa. [Here](https://stackoverflow.com/a/66546571), and [here](https://stats.stackexchange.com/a/236393) are two very impressive, highly recommended threads disscussing their relationships.

***

Learning rate is usualy the first thing to look at if your model performance no longer improves. Consider this exmaple. Let's suppose the true value of your parameter is 2.35, the current value of the parameter is 3. Let's suppose your learning rate is 0.5, meaning your parameter could only change in the multiple of 0.5. Your fist ```optimizer.step``` would update your parameter like this:

STEP 1  |  3 => 2.5 (target: 2.35)<br>
STEP 2  |  2.5 => 2.0 (target: 2.35)<br>
STEP 3  |  2.5 => 2.5 (target: 2.35)<br>
...

Learning rate of 0.5 is obviously too big! Although the parameter gradually stables at an acceptable value, if you want to further improve your model performance, in this exmaple, you might want to lower it down. Similarily, you would also want to increase your learning rate just so that your model parameter don't get stuck in a local minimum, or simply takes way too long to converge.

***

### Optimizer Selection

There are a lot of different optmisation algorithm available, [pytorch](https://docs.pytorch.org/docs/stable/optim.html) covers most of the mainstream ones, like [SGD](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html) and [AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html). There are a lot of other third party ones that are worth considering, [AdaFactor](https://huggingface.co/docs/transformers/main/en/main_classes/optimizer_schedules#transformers.Adafactor) which adjusts the learning rate adaptively, there are ones that targets devices with limited hardware like [LOMO](https://huggingface.co/papers/2306.09782), and my current faviouriate, [stable adamw](https://optimi.benjaminwarner.dev/optimizers/stableadamw/) that puts upper/ lower limit on how much parameter should update in one optimisation step.

***

### Scheduler

Scheduler controls how learning rate should change over time. If you understood a bit of MCMC you'd already be very familiar with the concept of learning rate schedulers. These essentially corresponds to the toggle that controls 'burn-in' or 'warm up' steps equivalent of MCMC procedure.

Schedulers for transformer models are much more intricate than the two stage process of burn-in/training in MCMC. Huggingface's [documentation](https://huggingface.co/docs/transformers/main/en/main_classes/optimizer_schedules#transformers.SchedulerType) of their learning rate schedulers provides an excellent overview of some of the available scheduler types, together with some very nice plots showing you how learning rates are adjusted. To name a few:

![MCMC](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_constant_schedule.png "[The warm-up-and-constant kind](https://huggingface.co/docs/transformers/main/en/main_classes/optimizer_schedules#transformers.get_constant_schedule_with_warmup)")

![coscine](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_cosine_schedule.png "[the coscine kind](https://huggingface.co/docs/transformers/main/en/main_classes/optimizer_schedules#transformers.get_cosine_with_hard_restarts_schedule_with_warmup)")

![coscine-restarts](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_cosine_hard_restarts_schedule.png "[The coscine with hard-restart kind](https://huggingface.co/docs/transformers/main/en/main_classes/optimizer_schedules#transformers.get_cosine_with_hard_restarts_schedule_with_warmup)")

```pytorch``` also comes with [even more](https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) learning rate schedulers to choose from. From my past experience, I found [warmup-stable-decay](https://huggingface.co/docs/transformers/main/en/main_classes/optimizer_schedules#transformers.get_wsd_schedule) (WSD) scheduler works the best. WSD scheduler works a bit like this:

1) **Warm up**: Learning rate slowly raises from a ```minimum``` to its ```maximum``` for a number of steps.
2) **Stable**: Learning rate stays at its ```maximum``` for another number of steps.
3) **Decay**: Learning rate slowly drops back down to its ```minimum``` for the rest of the training.

...But honestly I don't really know why WSD worked so well for my use case, guess it's probably just some usual LLM magic,,

### MISC

In addition to the main components listed above, there are also a handful of other things that would also be worth checking out: 

- How ```loss``` gets calculated, i.e., increase/ decrease the weight for certain class labels, use ```average``` or ```sum``` to calculate batch loss, etc. Some example configurations can be typically found at the documentation page of loss functions, [example](https://docs.pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html).
- Early stopping mechanism where the model training can stop early if certain metrics no longer imporve.
- If you were working with distributed training (the same model getting trained accross various devices), you would also need to consider configuring how different devices syncs with each other.
- Float point precision for your computation, basically how many decimal places your parameters are going to be. This sounds like a minor issue at first, but when you are working with gigabytes and gigabytes of computation, these things would significantly import your outcome.

***

## Practical advice on model training

### It's more about task management and less about math

...therefore it's not as hard as it appears to be. You don't need to have rich background in math/ physics to grasp good understading of transformers. 90% of works on model trianing is about building a good pipeline/ routine for managing it.

### Start with existing library

All these complication can be very intimidating and a big headache, thus of course there are a lot of very good python libraries that helps you manage all these ins and outs. I would strongly recommend you try out [Trainer](https://huggingface.co/docs/transformers/trainer) from the ```transformers``` library. ```transformers.Trainer``` comes with a set of defaults (as defined in ```transformers.TrainingArguments```) that already works very well for most of the situations out of the box, however you are still able to do a lot more customisations on your own should you wish to. Another thing that's very handy about ```Trainer``` is it comes with some pretty good advices in ```TrainingArguments``` [ API documentation page](https://huggingface.co/docs/transformers/v5.0.0/en/main_classes/trainer#transformers.TrainingArguments), including advices on distributed training too.

> Alothough ```transformers``` has received lots of critisms for being bloated with too many features over the years, I still think it is probably one of the easiest starting point for training transformer models. Furthermore, the [new V5](https://huggingface.co/blog/transformers-v5) version get rid of a lot of historical mess and things are definitely heading towards a much cleaner direction.

### You need to actively monitor model performance

Training a transformer can take days, throughout the process many things can go wrong. Always keep a good logging and always make sure checkpoins are saved with good and adequate intervals.

### Write clean codes!

![standards](https://imgs.xkcd.com/comics/standards.png "[standards](https://xkcd.com/927/)")

 Learning how to write clean APIs is probably the most crucial part of it. Your workload won't be tiny even considering all the packages out there claiming to be the one-stop solution of all the troubles. Every python installations already comes with their official advice on how to do it, just run:

```python
import this
```

Or in terminal, run:

```fish
python -m this
```

### Your code needs to be modular

You are 100% not going to have a good functional model after your very first model training. Things are guarentee to go wrong. You should be prepared to re-write your code over and over many times to get things right.

You will be adjusting a lot of different bits and pieces here and there in order to make things work. Therefore your code needs to be modular: you should be able to swap out part of your code without affecting the rest of it (i.e., model, scheduler, dataset, checkpointing, logging, the list goes on). And trust me, this is not an easy task.

As Phil Karlton once [said](https://www.karlton.org/2017/12/naming-things-hard/), "there are only two hard things in Computer Science: cache invalidation and naming things". Training transformer models, however, teaches you this *very* quickly.

## Further resources

- [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/), a very strong contender of ```transformers``` library. It mainly provides ```model.fit(dataset)``` api for ```pytorch``` models (similar to ```keras``` and ```sklearn```).
- [Huggingface's LLM course](https://huggingface.co/learn/llm-course/chapter0/1) on transformer models. This course contains a lot of very good conceptual guides on model training basics, together with very good step by step introductions on the API basics. Note that a lot of codes in the course are not really optimised for practical uses, but for the ease of understanding/ demonstration. You can certainly write much better codes than the example listed, once you were familar with its ins and outs.
- [Huge catelogue of notebook demos](https://huggingface.co/docs/trl/en/example_overview) on model training from ```TRL``` library. ```TRL``` is essentially like an 'extension' to huggingface's ```transformers``` library that focuses on much bigger model and distributed training. This catalogue of notebooks covers a lot of complicated model training scenarios, many of them can be a very good reference points if you want to build something more complicated, and happens to have access to much powerful hardwares.