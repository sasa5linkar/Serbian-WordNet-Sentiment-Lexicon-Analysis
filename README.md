# Serbian WordNet Sentiment Lexicon Analysis

Code, exported sentiment lexicons, and links to trained models accompanying:

> Petalinkar, Saša, Ranka M. Stanković, and Milica Ikonić Nešić (2025). **Comparative analysis of methods for creating a sentiment lexicon of the Serbian WordNet**. *The Electronic Library*, 43(4), 547–577. [doi:10.1108/EL-08-2024-0253](https://doi.org/10.1108/EL-08-2024-0253).

Use [CITATION.cff](CITATION.cff) for machine-readable citation metadata. This repository supports the sentiment-lexicon part of Saša Petalinkar's doctoral research.

## Start with the exported lexicons

For lookup or analysis, download an existing CSV below. Loading these files requires no model download, training, or API access. Each export contains **25,320 rows and 25,320 distinct synset IDs**.

| Lexicon | Method in the paper | Available resource / generation entry point |
| --- | --- | --- |
| S0 | SentiWordNet scores transferred through aligned WordNet synsets | External Serbian WordNet resource; see the S0 note below and [mappedLex.py](mappedLex.py) |
| S1 | SVM and Bernoulli Naive Bayes | [srbsentiwordnet1.csv](resources/srbsentiwordnet1.csv), [inference script](inferSVM_NB.py) |
| S2 | AdaBoost and Bernoulli Naive Bayes | [srbsentiwordnet2.csv](resources/srbsentiwordnet2.csv), [inference script](inferADA_NB.py) |
| S3 | RNN | [srbsentiwordnet3.csv](resources/srbsentiwordnet3.csv), [inference script](inferRNN.py) |
| S4 | Transformer | [srbsentiwordnet4.csv](resources/srbsentiwordnet4.csv), [inference script](inferTransformer.py) |
| S5 | BERTić | [srbsentiwordnet5.csv](resources/srbsentiwordnet5.csv), [model-based calculator](sentiwordnet_calculator.py) |
| S6 | GPT2-Orao (model IDs use `SRGPT`) | [srbsentiwordnet6.csv](resources/srbsentiwordnet6.csv), [model-based calculator](sentiwordnet_calculator.py) |
| S7 | Jerteh-355 | [srbsentiwordnet7.csv](resources/srbsentiwordnet7.csv), [model-based calculator](sentiwordnet_calculator.py) |

**S0 availability.** [mappedLex.py](mappedLex.py) reads transferred sentiment scores from `resources/wnsrp30.xml` and writes `resources/swn30_sentiment.csv`. These two files are not bundled. Obtain the Serbian WordNet XML separately from the resource holders; see the [Serbian WordNet service](https://wn.jerteh.rs/) for resource information.

### CSV fields

- `ID`: aligned WordNet synset identifier, e.g. `ENG30-03574555-n`. Use it for joins with the matching WordNet release.
- `POS` and `NEG`: numeric positive and negative sentiment scores. These are sentiment dimensions, not grammatical part-of-speech tags.
- `Unnamed: 0`: saved dataframe index in S1–S4; it is not a lexical identifier and may be ignored. S5–S7 contain only `ID,POS,NEG`.

Glosses and lemmas are **not columns in these seven exports**. Join them from a compatible, separately obtained lexical resource if needed. An objective component can be computed as `1 - POS - NEG`; it is not a stored column. Scores are between 0 and 1; POS and NEG sum to at most 1 within numerical tolerance.

### Load S5 with the Python standard library

Run from the repository root:

```python
import csv
from pathlib import Path

path = Path("resources/srbsentiwordnet5.csv")
with path.open(encoding="utf-8-sig", newline="") as stream:
    rows = list(csv.DictReader(stream))

lexicon = {
    row["ID"]: {"POS": float(row["POS"]), "NEG": float(row["NEG"])}
    for row in rows
}
assert len(rows) == len(lexicon) == 25320
scores = lexicon["ENG30-03574555-n"]
objective = 1.0 - scores["POS"] - scores["NEG"]
print(len(lexicon))
print(f"POS={scores['POS']:.6f} NEG={scores['NEG']:.6f} OBJ={objective:.6f}")
```

Expected output:

```text
25320
POS=0.001787 NEG=0.002019 OBJ=0.996194
```

## The 24 Hugging Face models

The three pretrained-model families each provide four **POS/NEG pairs**. `POS` predicts positive versus non-positive sentiment; `NEG` predicts negative versus non-negative sentiment. They classify Serbian synset definitions, rather than forming a single three-class classifier. Check each model card's label mapping before interpreting a returned confidence score.

The suffixes `0, 2, 4, 6` refer to training-set expansion iterations (T0, T2, T4, T6), **not training epochs or lexicon numbers S0–S7**.

| Lexicon / family | Iteration | POS model | NEG model |
| --- | --- | --- | --- |
| S5 / BERTić | T0 | [BERTicSENTPOS0](https://huggingface.co/Tanor/BERTicSENTPOS0) | [BERTicSENTNEG0](https://huggingface.co/Tanor/BERTicSENTNEG0) |
| S5 / BERTić | T2 | [BERTicSENTPOS2](https://huggingface.co/Tanor/BERTicSENTPOS2) | [BERTicSENTNEG2](https://huggingface.co/Tanor/BERTicSENTNEG2) |
| S5 / BERTić | T4 | [BERTicSENTPOS4](https://huggingface.co/Tanor/BERTicSENTPOS4) | [BERTicSENTNEG4](https://huggingface.co/Tanor/BERTicSENTNEG4) |
| S5 / BERTić | T6 | [BERTicSENTPOS6](https://huggingface.co/Tanor/BERTicSENTPOS6) | [BERTicSENTNEG6](https://huggingface.co/Tanor/BERTicSENTNEG6) |
| S6 / GPT2-Orao | T0 | [SRGPTSENTPOS0](https://huggingface.co/Tanor/SRGPTSENTPOS0) | [SRGPTSENTNEG0](https://huggingface.co/Tanor/SRGPTSENTNEG0) |
| S6 / GPT2-Orao | T2 | [SRGPTSENTPOS2](https://huggingface.co/Tanor/SRGPTSENTPOS2) | [SRGPTSENTNEG2](https://huggingface.co/Tanor/SRGPTSENTNEG2) |
| S6 / GPT2-Orao | T4 | [SRGPTSENTPOS4](https://huggingface.co/Tanor/SRGPTSENTPOS4) | [SRGPTSENTNEG4](https://huggingface.co/Tanor/SRGPTSENTNEG4) |
| S6 / GPT2-Orao | T6 | [SRGPTSENTPOS6](https://huggingface.co/Tanor/SRGPTSENTPOS6) | [SRGPTSENTNEG6](https://huggingface.co/Tanor/SRGPTSENTNEG6) |
| S7 / Jerteh-355 | T0 | [Jerteh355SENTPOS0](https://huggingface.co/Tanor/Jerteh355SENTPOS0) | [Jerteh355SENTNEG0](https://huggingface.co/Tanor/Jerteh355SENTNEG0) |
| S7 / Jerteh-355 | T2 | [Jerteh355SENTPOS2](https://huggingface.co/Tanor/Jerteh355SENTPOS2) | [Jerteh355SENTNEG2](https://huggingface.co/Tanor/Jerteh355SENTNEG2) |
| S7 / Jerteh-355 | T4 | [Jerteh355SENTPOS4](https://huggingface.co/Tanor/Jerteh355SENTPOS4) | [Jerteh355SENTNEG4](https://huggingface.co/Tanor/Jerteh355SENTNEG4) |
| S7 / Jerteh-355 | T6 | [Jerteh355SENTPOS6](https://huggingface.co/Tanor/Jerteh355SENTPOS6) | [Jerteh355SENTNEG6](https://huggingface.co/Tanor/Jerteh355SENTNEG6) |

In [sentiwordnet_calculator.py](sentiwordnet_calculator.py), let `p_pos` be the probability of the positive class from the POS model and `p_neg` the probability of the negative class from the NEG model. Each pair gives `POS = p_pos * (1 - p_neg)` and `NEG = p_neg * (1 - p_pos)`, with the remainder assigned to `OBJ`. The calculator averages the four iteration-specific pairs for each family. A model's predicted-label confidence must first be converted to the appropriate class probability.

## Repository map and training

| Location | Purpose |
| --- | --- |
| [resources/](resources/) | Published CSV lexicon exports and other resource files |
| [create_sets.py](create_sets.py) | Preparation and expansion of sentiment training sets |
| [mappedLex.py](mappedLex.py) | Export of mapped scores from the external Serbian WordNet XML |
| Inference scripts linked above | Archived paths for generating lexicons with the different methods |
| [sentiwordnet_calculator.py](sentiwordnet_calculator.py) | Paired Hugging Face inference and ensemble calculation |
| [environment.yml](environment.yml) | Archived research environment, including platform-specific dependencies |

The loading example uses a finished lexicon. For training, use the scripts above with the corresponding lexical resources, training sets and model dependencies. The WordNet XML and `train_sets/` directory must be supplied separately. Adapt paths and platform-specific dependencies in `environment.yml` to your environment.

The scores describe lexical senses. Sentence- or document-level sentiment analysis requires a separate method for selecting and combining senses. Mapped English scores provide the S0 baseline; the paper evaluates methods for enriching Serbian WordNet sentiment.

## License

The repository [LICENSE](LICENSE) is **CC0-1.0**. External lexical resources retain their own terms.

The eight BERTić sentiment models are licensed under **Apache-2.0**; the eight GPT2-Orao and eight Jerteh-355 sentiment models use **CC-BY-SA-4.0**. See the linked Hugging Face model cards for model-specific terms and base-model information. Please cite the accompanying paper when using these resources.
