---
title: "ELI5 Transformers (Part 2) - Generation"
date: 2025-11-11
publishDate: 2025-11-11

summary: "(it's glorified sequence classification)"
font_family: "Monospace"
tags: ["AI", "Machine Learning", "ELI5"]
topics: "transformers"
status: "draft"
draft: true
---

{{< katex >}}
![always has been](./featured.jpg "always has been")

*This is part 2 of the ELI5 transformers series, however you do not need to read Part 1 in order to follow the article. Link to the series can be found [here](https://yuyiheng.cc/topics/transformers/)*

I'm pretty sure you've already heard about something like this before: *'generative LLM is just slightly advanced auto complete'*. Or, something like *'predicting next word given all the previous words'*. Today I would like to invite you think of text generation from a slightly different perspective: <br>
**Text generation is sequence classification in a for-loop.** <br>
***
## Introduction
Let's use machine translation as our starting point. Ideally, our model would be something like:<br>
\(y = F(x)\)<br>
Where \(x\) is our input text, and \(y\) is our translated text.<br>

Such model won't be very hard for ```[digits, arabic numerals]```. It's just a look up table that converts one to another:
<style type="text/css">
.tg th{border-color:black;border-style:none;border-width:1px;
  padding:10px 25px;word-break:normal;}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
</style>
<table class="tg"><thead>
  <tr>
    <th class="tg-c3ow">digits</th>
    <th class="tg-c3ow">0</th>
    <th class="tg-c3ow">1</th>
    <th class="tg-c3ow">2</th>
    <th class="tg-c3ow">...</th>
  </tr></thead>
<thead>
  <tr>
    <th class="tg-c3ow">arabic</th>
    <th class="tg-c3ow">٠</th>
    <th class="tg-c3ow">١</th>
    <th class="tg-c3ow">٢</th>
    <th class="tg-c3ow">...</th>
  </tr>
</thead>
</table>

So our model would be:
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


Needless to say, it would be absurdly hard to do something similar for languages such as ```[English, French]``` or ```[Turkish, Spanish]```. Even if we do, what are the parameters for languages? Math works with numbers, but machine translation works with texts. Nowadays, almost all machine translations you've heard of (google, deepl etc.,) are done through the process of text-generation through LLMs.


***
***

I want to first introduce you with the idea of [universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem): we can see LLMs, or all transformer models as functions that are able to approximate the shape of any other functions. Suppose we have a target function, say, \(y = F(x)\), and we have our transformer \(y = T(x)\). With sufficient training, taking the same input \(x\), our transformer \(T\) is able to produce similar output \(y\) as the target function \(F\). In real life scenarios, we often don't have the formula for our target function \(F\), but, if we were only interested in getting an acceptable \(y\), then an approximation of \(F\) would be decent enough.<br>
