# BA-LR: Binary-Attribute based Likelihood Ratio estimation for speaker recognition

This library contains tools which can be used for explainable deep speaker recognition and modeling based on voice attributes. It is based on the [doctorate work](https://github.com/LIAvignon/BA-LR) of Imen Ben Amor but offers improved models and utilities for binary-attribute based speaker modeling.

The first step in the BA-LR approach for speaker recognition is the extraction of binary-attribute based speech representations. This can be done either with a model that works with raw audio files, or with a model pre-trained on speaker embeddings extracted using an embedding model, such as [wespeaker](https://github.com/wenet-e2e/wespeaker).
