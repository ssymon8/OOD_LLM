# Out-of-Domain Detection for Large Language Models

This work explores the behavior of Large Language Models (LLMs) when prompted with Out-of-Domain (OOD) queries. In this context, OOD is defined as data or concepts *not present in the model's training datasets*.

## Roadmap

For the moment, the project is structured around two main phases:

1. **Layer Activation Analysis:** We first analyze the reactions across the different hidden layers of the network when it processes unknown tasks or questions. The goal is to track the mechanics of hallucinations and observe the latent connections the neural network attempts to make.
2. **Mitigation Strategies:** We then explore and implement methods to detect and mitigate these behaviors, drawing inspiration from recent research papers :)

## Model & Tech Stack

For this project, I will be using **Ministral-3-3B-Instruct-2512** due to its lightweight footprint and ease of deployment.

## Disclaimer

This project is purely personal. It is built for exploration and learning, and has no intention of being used for commercial or formal research purposes.

## First Benchmarks

MMLU zero-shot : 0.6460
MMLU five-shot : 0.5381
