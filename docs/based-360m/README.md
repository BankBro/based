---
datasets:
- EleutherAI/pile
language:
- en
---
# Model Card

This model is pretrained Based model. Based is strong at recalling information provided in context, despite using a fixed amount of memory during inference.

As a quality reference, we include a pretrained Attention (Llama architecture) model provided here: https://huggingface.co/hazyresearch/attn-360m, and Mamba model provided here: https://huggingface.co/hazyresearch/mamba-360m

All three checkpoints are pretrained on **10Bn tokens** of the Pile in the exact same data order using next token prediction. 


### Model Sources

The model implementation and training code that produced the model are provided here: https://github.com/HazyResearch/based

### Uses

The purpose of this work is to evaluate the language modeling quality of a new efficient architecture, Based. 

We include a series of benchmarks that you can use to evaluate quality: 
- FDA: https://huggingface.co/datasets/hazyresearch/based-fda
- SWDE: https://huggingface.co/datasets/hazyresearch/based-swde
- SQUAD: https://huggingface.co/datasets/hazyresearch/based-squad


## Citation

Please consider citing this paper if you use our work: 

```
@article{arora2024simple,
  title={Simple linear attention language models balance the recall-throughput tradeoff},
  author={Arora, Simran and Eyuboglu, Sabri and Zhang, Michael and Timalsina, Aman and Alberti, Silas and Zinsley, Dylan and Zou, James and Rudra, Atri and Ré, Christopher},
  journal={arXiv:2402.18668},
  year={2024}
}
```

Please reach out to simarora@stanford.edu, eyuboglu@stanford.edu, and mzhang20@stanford.edu with questions.
