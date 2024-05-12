import pytest
import torch
from torch import nn
from models.audio import audiomae_ctx8, mel128, mn04, mn20, openl3_mel256_music_6144, passt_s


@pytest.mark.parametrize(
    # "audio_model", [audiomae_ctx8, mel128, mn04, mn20, openl3_mel256_music_6144, passt_s]
    "audio_model",
    [mel128, mn04, mn20, openl3_mel256_music_6144],
)
def test_output_shape(audio_model: nn.Module):
    audio_model = audio_model()
    audio_model.eval()
    input_data = torch.randn((32, audio_model.in_channels, audio_model.sample_rate))
    assert audio_model(input_data).shape == (32, audio_model.out_features)
