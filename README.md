"""
# Real-Time Multimodal Turn-taking Prediction to Enhance Cooperative Dialogue during Human-Agent Interaction

## Authors
- Young-Ho Bae
- Casey C. Bennett, Member, IEEE

## Conference Details
- **Conference**: 2023 32nd IEEE International Conference on Robot and Human Interactive Communication (RO-MAN)
- **Date**: August 28-31, 2023
- **Venue**: Paradise Hotel, Busan, Korea

## Abstract
Predicting when it is an artificial agent’s turn to speak/act during human-agent interaction (HAI) poses a significant challenge due to the necessity of real-time processing, context sensitivity, capturing complex human behavior, effectively integrating multiple modalities, and addressing class imbalance. In this paper, we present a novel deep learning network-based approach for predicting turn-taking events in HAI that leverages information from multiple modalities, including text, audio, vision, and context data. Our study demonstrates that incorporating additional modalities, including in-game context data, enables a more comprehensive understanding of interaction dynamics leading to enhanced prediction accuracy for the artificial agent. The efficiency of the model also permits potential real-time applications. We evaluated our proposed model on an imbalanced dataset of both successful and failed turn-taking attempts during an HAI cooperative gameplay scenario, comprising over 125,000 instances, and employed a focal loss function to address class imbalance. Our model outperformed baseline models, such as Early Fusion LSTM (EF-LSTM), Late Fusion LSTM (LF-LSTM), and the state-of-the-art Multimodal Transformer (Mult). Additionally, we conducted an ablation study to investigate the contributions of individual modality components within our model, revealing the significant role of speech content cues. In conclusion, our proposed approach demonstrates considerable potential in predicting turn-taking events within HAI, providing a foundation for future research with physical robots during human-robot interaction (HRI).

## Code Structure
- **modules**: Contains helper scripts for the model architecture.
    - `crossmodal_transformer.py`: Implementation of the crossmodal transformer.
    - `model_architecture.py`: Defines the overall architecture of the model.
    - `multihead_attention.py`: Implementation of the multihead attention mechanism.
    - `positional_encoding.py`: Provides the positional encoding for the transformer architecture.
- **train.ipynb**: Jupyter notebook to train the model.

## Usage Instructions
To train the model, execute the `train.ipynb` notebook. Ensure all dependencies are installed and data paths are set appropriately. 

## Acknowledgements
(To be added, if any)

## License
(To be added, if any)
"""
