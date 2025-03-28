### README.md 翻译

# 训练基于模型

为了使用我们的代码训练新模型，你需要完成一些额外的设置：

```python
# 安装训练所需的额外依赖项
pip install -e .[train]

# 安装 Apex（如果遇到问题，可能是由于 torch 或 pip 版本问题；如果你使用的是 torch 2.0.1，可以参考 https://github.com/NVIDIA/apex/issues/1735）
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
cd ..
```

我们将本节分为三个部分：1）如何设置训练配置并启动；2）如何设置快速训练内核；3）如何安装额外的训练优化。

---

### 启动训练
要训练一个新模型，请在 `train/configs/experiment/` 目录下创建一个 `config.yaml` 文件。我们提供了用于生成论文中预训练检查点的配置文件（发布在 Hugging Face 上），位于 `train/configs/experiment/reference/`。

你可以通过以下命令从 `train/` 目录启动训练任务，其中可以修改配置名称和 GPU 数量（`trainer.devices`）：
```
cd train/
python run.py experiment=reference/based-1b trainer.devices=8
```

在我们的论文中，我们在 Pile 数据集上进行了评估，但该数据集已不再在线提供，因此 `train/configs/experiment/reference/` 中的配置文件无法直接运行。为此，我们提供了一个示例配置文件，用于在 WikiText103 语言建模数据集上进行训练。你可以使用以下脚本启动训练：
```
cd train/
python run.py experiment=example/based-360m trainer.devices=8
```

你可以通过在 `train/configs/datamodule/` 下添加新的数据集配置文件来适配训练数据集。请参考 `wikitext103.yaml` 中的示例。创建新的数据集 YAML 文件后，进入实验配置文件（例如 `train/configs/experiment/example/based-360m.yaml`），将 `override datamodule` 下的数据模块名称更新为你的新数据集 YAML 文件名。

在启动训练之前，请务必更新配置文件中的检查点目录 [在此处](https://github.com/HazyResearch/based/blob/3fb009b8216b41d14ea3a2ab9552a5c609ef0bf4/train/configs/experiment/example/based-360m.yaml#L39)。

---

### 快速训练
我们在本仓库中支持几种不同的训练视图。训练配置中的 `parallel_implementation` 参数决定了使用哪种训练视图：
[链接](https://github.com/HazyResearch/based/blob/e86e21401ad26e38a46590e73af43868f4a98b2a/based/models/mixers/linear_attention.py#L73)

默认情况下，无需安装任何内核，训练时会保留二次复杂度 O(n²) 的视图。目前我们推荐使用下面的选项 2 来显著加速训练。这些方法将被替换为我们即将发布的自定义内核（来自 Based 论文）。

- **选项 1 (`parallel_implementation = "quadratic"`)**：默认值，使用 PyTorch 的二次复杂度视图。
- **选项 2 (`parallel_implementation = "fla_parallel"`)**：Flash Linear Attention 内核。使用以下命令安装：
  ```
  pip install triton==2.2.0
  pip install -U git+https://github.com/sustcsonglin/flash-linear-attention
  ```
- **选项 3 (`parallel_implementation = "linear"`)**：Fast Transformers 的线性注意力内核。使用以下命令安装：
  ```
  cd train/csrc/causal_dot_prod/
  python setup.py install
  ```

我们在 [benchmark/examples/linear_attention_forward/](https://github.com/HazyResearch/based/tree/main/benchmark) 文件夹中提供了不同内核的基准测试结果。此外，我们还提供了 [WandB 训练曲线](https://api.wandb.ai/links/simarora/ryv84b55)，展示了使用 `fla-parallel` 模式如何让 Based 在保持高质量的同时快速训练！

---

### 其他注意事项
- **其他融合操作的内核**：  
  默认情况下，配置文件会使用来自 [Flash Attention](https://github.com/Dao-AILab/flash-attention) 仓库的融合内核，可以通过克隆该仓库并运行 `python setup.py install` 来安装相关内核（如 `fused_dense_lib`、`layer_norm`、`rotary` 和 `xentropy`）。或者，你可以通过以下方式避免使用这些内核：
  - 在实验配置文件中指定 `fused_dense=False`。
  - 将 `based/models/gpt.py` 中的 `RMSNorm` 导入路径替换为 `based/ops/triton/layer_norm`。

- **衰减策略**：  
  如果你想探索论文中提到的可选衰减策略，可以查看 [notebooks/03-31-decay.ipynb](https://github.com/HazyResearch/based/blob/main/notebooks/03-31-decay.ipynb) 笔记本。

- **引用说明**：  
  请注意，本训练代码来源于以下项目：
  - [Flash Attention](https://github.com/Dao-AILab/flash-attention/tree/main/training)
  - Flash Linear Attention 内核来源于 [https://github.com/sustcsonglin/flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention)
  - Fast Transformers 内核来源于 [https://github.com/idiap/fast-transformers](https://github.com/idiap/fast-transformers)  
  **如果你使用了他们的工作，请务必引用！**

--- 

### 总结
该 README 文件详细介绍了如何使用 **Based** 项目进行模型训练，包括环境配置、训练启动、性能优化以及注意事项。通过选择合适的内核和配置，可以显著提升训练效率。同时，项目还提供了丰富的参考资料和工具来源，方便用户进一步探索和扩展。