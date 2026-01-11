---
title: "Transformer"

cascade:
  showDate: false
  showAuthor: false
  showSummary: true
  showViews: false
  showLikes: false
  invertPagination: false
  orderByWeight: true
---
As someone without much backrounds in neither physics nor computer science, I find lots of available introductions on transformers very confusing. However, most of the articles on transformers focuses on attention mechanisms, using either [the OG transformer](https://peterbloem.nl/blog/transformers) or [the classic BERT](https://towardsdatascience.com/a-complete-guide-to-bert-with-code-9f87602e4a11/) as examples. There are a lot of very good learning materials out there, for example the amazing [interactive transofmer explainer](https://poloclub.github.io/transformer-explainer).<br>

Finally, I've decided to bite the bullet and spend some time have a read through the HuggingFace's source code for many of these models, [like this one](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py). I took lots of notes here and there during the process of studying transformers.<br>

This series of blogposts collects my notes and ELI5s of what I've learned, hope these notes can help others alongside their studying, or being interesting little pieces of articles to read through.
***