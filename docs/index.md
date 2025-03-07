
<script type="text/x-mathjax-config"> MathJax.Hub.Config({ TeX: { equationNumbers: { autoNumber: "all" } } }); </script>
<script type="text/x-mathjax-config">
	MathJax.Hub.Config({
		tex2jax: {
			inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      processEscapes: true
  }
});
</script>
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<!-- ... -->

<link href="https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css" rel="stylesheet" integrity="sha384-wvfXpqpZZVQGK6TAh5PVlGOfQNHSoD2xbE+QkPxCAFlNEevoEH3Sl0sibVcOQVnN" crossorigin="anonymous" />
<link rel="stylesheet" href="{{ site.baseurl}}/css/trackswitch.min.css" />


# Abstract
Deep learning appears as an appealing solution for Automatic Synthesizer Programming (ASP), which aims to assist musicians and sound designers in programming sound synthesizers. However, integrating software synthesizers into training pipelines is challenging due to their potential non-differentiability. This work tackles this challenge by introducing a method to approximate arbitrary synthesizers. Our method trains a neural network to map synthesizer presets onto a perceptually informed embedding space defined by a pretrained audio model. This process creates a differentiable neural proxy for a synthesizer by leveraging the audio representations learned by the pretrained model. We evaluate the representations learned by various pretrained audio models in the context of neural-based nASP and assess the effectiveness of several neural network architectures – including feedforward, recurrent, and transformer-based models – in defining neural proxies. We evaluate the proposed method using both synthetic and hand-crafted presets from three popular software synthesizers and assess its performance in a synthesizer sound matching downstream task. Encouraging results were obtained for all synthesizers, paving the way for future research into the application of synthesizer proxies for neural-based ASP systems.

# Sound Matching Examples
To demonstrate the integration of a preset encoder into an nASP pipeline and assess its potential benefits, we evaluated its performance to on a Synthesizer Sound Matching (SSM) downstream task using Dexed and Diva, on both in-domain and out-of-domain sounds. 

The objective of the SSM task was to infer the set of synthesizer parameters that best approximates a given target audio. The preset encoder is used to compute a perceptual loss between its representation of the predicted preset and the representation of the target audio produced by a pretrained audio model (here we used `mn20` from [EfficientAT](https://github.com/fschmid56/EfficientAT_HEAR/)). 


We evaluated three different loss scheduling configurations to assess the benefit of adding a perceptual loss, inspired by [Masuda & Saito (2023)](https://github.com/hyakuchiki/SSSSM-DDSP):
- `PLoss`. Only the parameter loss is used and serves as a baseline.
- `Mix`. The parameter loss is applied for the first 200 epochs, then the perceptual loss is gradually introduced over the next 200 epochs, and the estimator network is trained for the remaining 200 epochs using both parameter and perceptual losses.
- `Switch`. This approach is similar to the previous one, but fully transitions from parameter loss to perceptual loss, resulting in the estimator network being trained exclusively with perceptual loss during the final 200 epochs.

## Out-of-domain sounds
The following audio target examples are taken from the validation set of the [NSynth dataset](https://magenta.tensorflow.org/datasets/nsynth) (scroll right to access the other examples).

<table>
  <thead>
    <tr>
      <th></th>
      <th>Bass elec.</th>
      <th>Bass synth.</th>
      <th>Brass ac.</th>
      <th>Flute synth.</th>
      <th>Guitar elec.</th>
      <th>Keyboard elec.</th>
      <th>Mallet ac.</th>
      <th>Organ elec.</th>
      <th>Reed ac.</th>
      <th>String ac.</th>
      <th>Vocal ac.</th>
    </tr>
  </thead>
  <tbody>
    {% for model in site.models %}
    <tr>
      <td>{{ site.model_names[model] }}</td>
      {% for source in site.sources %}
      <td><audio src="{{ site.baseurl }}/assets/audio/nsynth/{{ source }}_{{ model | downcase | replace: ' ', '-' }}.wav" controls></audio></td>
      {% endfor %}
    </tr>
    {% endfor %}
  </tbody>
</table>
