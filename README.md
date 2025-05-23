<div align="center">

# 🎬 Fostering Video Reasoning via Next-Event Prediction


<div>
🚀  Toward Video Reasoning via Future Prediction 🌟
</div>
</div>
<div>
<br>

<div align="center">

[![Github](https://img.shields.io/badge/|V1-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/haonan3/V1)
[![Notion](https://img.shields.io/badge/|Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white)](https://v1-videoreasoning.notion.site/) 
[![Twitter](https://img.shields.io/badge/V1-white?style=for-the-badge&logo=X&logoColor=000&color=000&labelColor=white)](https://x.com/Haonan_Wang_/status/1901684827621072933)
[![Hugging Face Collection](https://img.shields.io/badge/|_Dataset_V1_33K-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor)](https://huggingface.co/datasets/haonan3/V1-33K)

</div>

</div>


---

Welcome to the official repository for **Fostering Video Reasoning via Next-Event Prediction**! 🚀  
Read our paper on arXiv: [📖 2411.13476](https://arxiv.org/pdf/2411.13476)  
Browse the dataset on Hugging Face: [📂 V1-33K](https://huggingface.co/datasets/haonan3/V1-33K)


---

## Video Reasoning via Future Prediction

To advance multimodal LLMs' reasoning ability, we introduce a future prediction task and its corresponding dataset. Predicting upcoming events from historical video data presents significant challenges for current multimodal LLMs. Our task pushes these models to infer future events based on the first part of a video, with the second part serving as open-ended ground truth (Self-Supervised Learning).

> **🤔 <ins>Why isn’t factual answering ideal for video reasoning?</ins>**  
> Research indicates that reasoning models like DeepSeek R1 often “over-think”, which can lead to hallucinations. When applied to video data, similar pitfalls emerge if the model is restricted to answering straightforward factual questions. For instance, querying “Where is the cat in the video?” might prompt an overly extended reasoning process, inadvertently increasing the risk of hallucinated outputs.

> **💡 <ins>Why is future prediction a compelling case for video reasoning?</ins>** <a id="why-video-prediction"></a>   
> Much like Doctor Strange’s foresight in `Avengers 3: Infinity War (2018)`, predicting the future demands reasoning over multiple potential outcomes. This challenge is analogous to techniques such as Monte Carlo tree search (MCTS), which systematically explores a wide array of possible scenarios. The inherent complexity of future prediction makes it a powerful task for evaluating and enhancing video reasoning capabilities.  
>  
> ![assets/example.png](assets/example.png)

> **📽️ <ins>Video Future Prediction: A Self-Supervised Task for Multimodal Reasoning</ins>**  
> This task is inherently Self-Supervised Learning (SSL). It leverages the inherent causal logic present in video data. By dividing videos into sequential segments, we create implicit labels that embody the natural flow of cause and effect—allowing models to learn from the logical progression of events *without* the need for manual annotations.  
>  
> Much like `Image Contrastive Learning`, which uses inherent data structures to construct labels and guide what a model should capture, `Video Future Prediction` is grounded in the philosophy that real-world events unfold through a chain of cause and effect. It drives the model to focus on the temporal and causal dimensions that underpin real-world scenarios, enhancing multimodal reasoning capabilities. By integrating visual cues, the model develops a holistic reasoning ability to more accurately predict and interpret the progression of complex events.  
>  
> Moreover, like other self-supervised learning tasks and unsupervised learning, the data construction is relatively cheap, making it a scalable solution for enhancing multimodal reasoning capabilities.





---

## 📦 Features

- 🔍 **Next-Event Prediction** for video reasoning  
- 🎓 Demo scripts for instruction tuning & reinforcement learning  
- 🛠️ Easy use with [LLaMA-Factory on GitHub](https://github.com/hiyouga/LLaMA-Factory) & [EasyR1](http://github.com/hiyouga/EasyR1)  

---

## 🐍 Setup

### 1. Create a Conda environment  
```bash
conda create -n video_llm python=3.10 -y
conda activate video_llm
````

### 2. Download the V1-33K dataset

```bash
python v1_data_download.py
```

> You should now see a folder named `V1-33K/` containing:
>
> * `first_part_video/`
> * `video_dataset/`

---

## 🔧 LLaMA-Factory Integration

1. **Clone the repo**

   ```bash
   git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
   cd LLaMA-Factory
   ```

2. **Install dependencies**

   ```bash
   pip install -e ".[torch,metrics]" --no-build-isolation
   ```

---

## 🗄️ Preparing Next-Event Prediction Data

```bash
# From the project root
python video_data_generation.py
```

> The generated data will be placed in `./LLaMA-Factory/data/`

### Move necessary files

```bash
mv dataset_info.json LLaMA-Factory/data/
mv qwen2_5vl_7B_full_sft_5K.yaml LLaMA-Factory/examples/train_full/
```

---

## 🚀 Demo Training

* **Instruction Tuning**

  ```bash
  bash video_instruction_tuning_demo.sh
  ```

---

## 🤖 Reinforcement Learning with GRPO

1. **Install RL Env**
   ```bash
   cd EasyR1
   pip install -e .
   ```

2. **Run the GRPO training demo**
   ```bash
   bash video_GRPO_training_demo.sh
   ```

---


## 🔥 Evaluation

We run all our evaluations based on the `lmms-eval`. Besides those benchmarks that have been implemented in `lmms-eval`, we also incorporate evaluations of our `FutureBench` as well as `SeedBench-R1` into it. To start,

1. **Install lmms-eval**
   ```bash
   # eval with lmms-eval
   cd third_party/lmms-eval
   pip install -e .
   ```

2. **Preparing Dataset**
   > You should also find the `futurebench.json` under the same folder named `V1-33K/`.
   
   ```bash
   # make dataset from futurebench.json 
   python gen_dataset.py
   ```

3. **Run the inference**
  
   > Before running the following eval script, check the `dataset_path` and `cache_dir` in `third_party/lmms-eval/lmms_eval/tasks/futurebench/futurebench.yaml` are correct.

   ```bash
   bash third_party/lmms-eval/examples/eval_futurebench.sh
   ```

   > To run evaluations on other benchamarks, see more settings in `third_party/lmms-eval/examples/`.

---

## 📚 Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{wang2024precision,
  title={When Precision Meets Position: BFloat16 Breaks Down RoPE in Long-Context Training},
  author={Wang, Haonan and Liu, Qian and Du, Chao and Zhu, Tongyao and Du, Cunxiao and Kawaguchi, Kenji and Pang, Tianyu},
  journal={arXiv preprint arXiv:2411.13476},
  year={2024}
}
```

---

😊 Happy exploring & feel free to open an issue or pull request! 🎉


