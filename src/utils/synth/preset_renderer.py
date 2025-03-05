"""
Module used to render VST3 presets from arbitrary synths.
"""

from pathlib import Path
from typing import List, Sequence
import dawdreamer as daw
import numpy as np


class PresetRenderer:
    """
    Class to rendered audio presets using DawDreamer.
    """

    def __init__(
        self,
        synth_path: str,
        sample_rate: int = 44_100,
        render_duration_in_sec: float = 4,
        fadeout_in_sec: float = 0.1,
        convert_to_mono: bool = True,
        normalize_audio: bool = False,
    ):
        ### Paths and name related member variables
        self.synth_path = synth_path
        self.synth_name = Path(self.synth_path).stem

        ### DawDreamer related member variables
        self.sample_rate = sample_rate
        self.render_duration_in_sec = render_duration_in_sec
        self.engine = daw.RenderEngine(self.sample_rate, block_size=128)  # pylint: disable=E1101
        self.synth = self.engine.make_plugin_processor(self.synth_name, self.synth_path)
        graph = [(self.synth, [])]
        self.engine.load_graph(graph)

        ### MIDI related member variables
        self.midi_note: int = None
        self.midi_velocity: int = None
        self.midi_duration_in_sec: float = None

        ### Rendering related member variables
        # fadeout
        self.fadeout_in_sec = fadeout_in_sec
        self._fadeout_len = int(self.sample_rate * self.fadeout_in_sec)
        # avoid multiplication with an empty array (from linspace) if fadeout_in_sec \neq 0
        if self._fadeout_len > 0:
            self._fadeout = np.linspace(1, 0, self._fadeout_len)
        else:  # hard-coding if fadeout_duration_s = 0
            self._fadeout = 1.0

        # export to mono
        self.convert_to_mono = convert_to_mono

        # normalize
        self.normalize_audio = normalize_audio

    @property
    def midi_params(self):
        """
        Get the instance's midi note.

        Returns: midi note as tuple (note: int, velocity: int, duration_in_sec: float).
        """
        return (self.midi_note, self.midi_velocity, self.midi_duration_in_sec)

    def set_parameters(self, params_idx: Sequence[int], params_val: Sequence[float]) -> None:
        """
        Args
        - `params_idx` (List[int]): List of indices for each parameter.
        - `params_val` (List[float]): List of values for each parameter.
        """
        # individually set each parameters since DawDreamer does not accept
        # list of tuples (param_idx, value)
        for idx, val in zip(params_idx, params_val):
            self.synth.set_parameter(idx, val)

    def get_parameters(self, params_idx: Sequence[int]) -> List[float]:
        """
        Args
        - `params_idx` (Optional[List[int]]): indices of the parameters to retrieve.
        """
        return [self.synth.get_parameter(idx) for idx in params_idx]

    def set_midi_parameters(
        self,
        midi_note: int = 60,
        midi_velocity: int = 100,
        midi_duration_in_sec: float = 1.0,
    ):
        """Set the instance's midi note."""
        self.midi_note = midi_note
        self.midi_velocity = midi_velocity
        self.midi_duration_in_sec = midi_duration_in_sec

    def render_note(self) -> Sequence:
        """
        Renders a midi note (for the currently set patch) and returns the generated audio as ndarray.
        """
        if self.midi_note is None:
            raise ValueError("No midi note has been set yet. Please use `set_midi_note()` first.")

        # graph = [(self.synth, [])] # done once when initializing the class
        # self.engine.load_graph(graph) # done once when initializing the class
        self.synth.add_midi_note(60, 0, 0, 0.01)
        self.engine.render(0.01)  # FIXME: to clear audio buffer  (for long release, delay, etc.)
        self.synth.add_midi_note(self.midi_note, self.midi_velocity, 0.01, self.midi_duration_in_sec)
        self.engine.render(self.render_duration_in_sec)
        self.synth.clear_midi()
        audio = self.engine.get_audio()
        if self.convert_to_mono:
            audio = np.mean(audio, axis=0, keepdims=True)
        if self.normalize_audio:
            audio = audio / np.max(np.abs(audio))
        audio[..., -self._fadeout_len :] = audio[..., -self._fadeout_len :] * self._fadeout  # fadeout
        return audio


if __name__ == "__main__":
    print("breakpoint me!")
