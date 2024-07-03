# legalhatespeech
Code for the paper "Legally Enforceable Hate Speech Detection for Public Forums" published in Findings of EMNLP 2023 ([paper link](https://aclanthology.org/2023.findings-emnlp.730.pdf))


# Generating silver data
1. Sample easy negatives using `dataset/sampling negatives.ipynb`
2. Generate the labels with `silver label generation.ipynb`
3. Unify the natural text generations with legal annotations with `silver data refactor.ipynb`
4. (Optional) Run the notebook `data/construct final dataset.ipynb` to generate prompt files into the directories `data/train` and `data/test`.

# Zero-shot experiments
All code is available under `zeroshot`. This requires the prepared data, or you can use the prompt formatting directly from `data/construct final dataset.ipynb`.

# Self-training
Run the notebook `data/construct final dataset.ipynb` to generate prompt files into the directories `data/train` and `data/test`. This notebook requires the data files.

# Contact and citations

If you are using this code or our data, please cite our EMNLP paper:
```
@inproceedings{luo-etal-2023-legally,
    title = "Legally Enforceable Hate Speech Detection for Public Forums",
    author = "Luo, Chu Fei and
      Bhambhoria, Rohan  and
      Dahan, Samuel  and
      Zhu, Xiaodan",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.730",
    doi = "10.18653/v1/2023.findings-emnlp.730",
    pages = "10948--10963",
}
```

Please email chufei.luo@queensu.ca for the data. We are considering uploading the dataset to Huggingface, but this is still TBD. Thank you!
