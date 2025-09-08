## Adding audio models
1. Create a `src/models/audio/<new_model>` directory and copy paste the model's source code into it.

2. Create a wrapper for the model inheriting from the `AudioModel` abstract class available in [src/models/audio/abstract_audio_model.py](./src/models/audio/abstract_audio_model.py), and implement the necessary methods.

3. Instantiate the model below the wrapper class (see existing models for example). 

4. Add the instantiated model to the models.audio package by adding it in [src/models/audio/\_\_init\_\_.py](./src/models/audio/__init__.py).

Once these steps are completed, the model can be used to generate a new dataset using the argument `audio_fe=<my_model>`. See [Data Generation](#data-generation) for more details.

→ The current implementation only allow the use of 1D (i.e., reduced) audio embeddings.