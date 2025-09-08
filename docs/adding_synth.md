# Adding a new synthesizer
To add a new synthesizer, follow these steps:

1. Make sure the synthesizer is supported by [DawDreamer](https://github.com/DBraun/DawDreamer).

2. Add its path to the `.env` file, e.g., `NEW_SYNTH_PATH=/path/to/new/synth/vst3`.
3. Create a file under [src/data/synths/](./src/data/synths/) that defines the synthesizer as a tuple of `SynthParameter` instances.
   * Use [src/data/synths/another_synth.py](./src/data/synths/another_synth.py) as a template.
   * Use [src/utils/synth/get_synth_parameters.py](./src/utils/synth/get_synth_parameters.py) to auto-generate parameter descriptions (then double-check).

4. Add additional arguments for each SynthParameter instance if desired (see [src/utils/synth/synth_parameter.py](./src/utils/synth/synth_parameter.py) and existing synthesizers for examples). These are used to constraint the sampling process used to generated synthetic presets.

5. Register the synth:
   * Add the python module for the new synthesizer to the package under [src/data/synths/\_\_init\_\_.py](./src/data/synths/__init__.py).
   * Add the synthesizer name
     * to the list of supported synthesizer names in the the `PresetHelper` class definition in [src/utils/synth/preset_helper.py](./src/utils/synth/preset_helper.py), and 
     * in the `SynthDataset` class definition in [src/data/datasets/synth_dataset.py](./src/data/datasets/synth_dataset.py).

6. Create a configuration file under [./configs/export/synth](./configs/export/synth) for the synthesizer specifying the parameters to exclude. The excluded parameters will be set to their default values during sampling and will not be fed to the preset encoder. 

7.  Once the datasets have been generated, files in the following configuration folders need to be added depending on the need (see existing synthesizers for examples).