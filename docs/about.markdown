---
layout: page
title: Neural Proxies for Sound Synthesizers
permalink: /about/
---

#  Neural Proxies for Sound Synthesizers: Learning Perceptually Informed Preset Representations

## Abstract
Deep learning appears as an appealing solution for Automatic Synthesizer Programming (ASP), which aims to assist musicians and sound designers in programming sound synthesizers. However, integrating software synthesizers into training pipelines is challenging due to their potential non-differentiability. This work tackles this challenge by introducing a method to approximate arbitrary synthesizers. Our method trains a neural network to map synthesizer presets onto a perceptually informed embedding space defined by a pretrained audio model. This process creates a differentiable neural proxy for a synthesizer by leveraging the audio representations learned by the pretrained model. We evaluate the representations learned by various pretrained audio models in the context of neural-based nASP and assess the effectiveness of several neural network architectures – including feedforward, recurrent, and transformer-based models – in defining neural proxies. We evaluate the proposed method using both synthetic and hand-crafted presets from three popular software synthesizers and assess its performance in a synthesizer sound matching downstream task. Encouraging results were obtained for all synthesizers, paving the way for future research into the application of synthesizer proxies for neural-based ASP systems.