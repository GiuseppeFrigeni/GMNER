# CV Project

In this repo there are the code and the notebook used for the Project for the exam of Computer Vision 2024-2025. The chosen topic is based on the paper [Grounded Multimodal Named Entity Recognition on Social Media](https://aclanthology.org/2023.acl-long.508.pdf).

## Requrirements
There are no special requirements for Kaggle as can be seen on the notebook as of June 2025.

##  Main objectives

The main objectives have been carried out to completion with these modalities:

- Plenty of work has been done on this repo to make it work without the need of outdated/abandoned libraries like transformers 3.4.0 and fastnlp, incorporating some functionalities directly in the repo and updating others.

- The ablation study has been done in two ways, first by passing zeros as features for images and secondly changing the architecture to not consider images at all and we obtained similar results.

- To compress the model two modalities were tried, Quantization Aware Training which provided bad performances and Dynamic Quantization that kept the performance very close to original while halving the size of the model.
