# Alpha Alternator

## Introduction
The **Alpha Alternator** is a novel generative model designed for time-dependent data, dynamically adapting to the complexity introduced by varying noise levels in sequences. Unlike state-of-the-art dynamical models such as Mamba, which assume uniform noisiness across sequences, the Alpha Alternator utilizes the **Vendi Score (VS)** to adjust the influence of sequence elements on predicted future dynamics at each time step.

![The $\alpha$-Alternator is robust to varying noise levels compared to a Mamba and an Alternator. The Alternator is more robust to noise than the Mamba.](./assets/figure_1.png)

## Key Features
- **Dynamic Noise Adaptation**: Adjusts reliance on input sequences versus latent history based on a learned parameter.
- **Vendi Score (VS) Integration**: Uses a similarity-based diversity metric to determine the informativeness of sequence elements.
- **Alternator Loss Minimization**: Optimizes model robustness through a combination of observation masking and loss minimization.
- **Robustness to Noisy Data**: Learns to differentiate between noisy and informative sequence elements for improved predictions.
- **Superior Performance**: Outperforms Alternators and state-space models in **trajectory prediction, imputation, and forecasting** tasks.

## Methodology
The Alpha Alternator adjusts its prediction strategy based on a learned parameter:
- **Negative Parameter**: Indicates a noisy dataset; the model prioritizes latent history over individual sequence elements that increase VS.
- **Positive Parameter**: Suggests an informative dataset; the model prioritizes new inputs that increase VS over latent history.

Training involves **observation masking** to simulate varying noise levels and **Alternator loss minimization** to enhance robustness.

## Installation
To set up the environment for running the Alpha Alternator model, install the necessary dependencies:
```bash
pip install -r requirements.txt
```


## Results
Our experimental results show that the Alpha Alternator achieves state-of-the-art performance in **neural decoding and time-series forecasting** benchmarks, surpassing existing Alternators and state-space models.

## Citation
If you use the Alpha Alternator model in your research, please cite:
```bibtex
@article{rezaei2025alpha,
  title={The Alpha-Alternator: Dynamic Adaptation To Varying Noise Levels In Sequences Using The Vendi Score For Improved Robustness and Performance},
  author={Rezaei, Mohammad Reza and Dieng, Adji Bousso},
  journal={arXiv preprint arXiv:2502.04593},
  year={2025}
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.


