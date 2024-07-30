from pathlib import Path


import pytest
import torch
from unittest.mock import MagicMock
from litgpt.api import LLM, calculate_number_of_devices
from litgpt.scripts.download import download_from_hub
from tests.conftest import RunIf


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLM)
    llm.model = MagicMock()
    llm.preprocessor = MagicMock()
    llm.prompt_style = MagicMock()
    llm.checkpoint_dir = MagicMock()
    llm.fabric = MagicMock()
    return llm


def test_load_model(mock_llm):
    assert isinstance(mock_llm, LLM)
    assert mock_llm.model is not None
    assert mock_llm.preprocessor is not None
    assert mock_llm.prompt_style is not None
    assert mock_llm.checkpoint_dir is not None
    assert mock_llm.fabric is not None


def test_generate(mock_llm):
    prompt = "What do Llamas eat?"
    mock_llm.generate.return_value = prompt + " Mock output"
    output = mock_llm.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=5)
    assert isinstance(output, str)
    assert len(output) > len(prompt)


@RunIf(min_cuda_gpus=1)
def test_quantization_is_applied(tmp_path):
    download_from_hub(repo_id="EleutherAI/pythia-160m", tokenizer_only=True, checkpoint_dir=tmp_path)
    llm = LLM.load("EleutherAI/pythia-160m", quantize="bnb.nf4", init="random", tokenizer_dir=Path(tmp_path/"EleutherAI/pythia-160m"))
    assert "NF4Linear" in str(type(llm.model.lm_head))


def test_stream_generate(mock_llm):
    prompt = "What do Llamas eat?"

    def iterator():
        outputs = (prompt + " Mock output").split()
        for output in outputs:
            yield output

    mock_llm.generate.return_value = iterator()
    output = mock_llm.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=5, stream=True)
    result = "".join([out for out in output])
    assert len(result) > len(prompt)


def test_generate_token_ids(mock_llm):
    prompt = "What do Llamas eat?"
    mock_output_ids = MagicMock(spec=torch.Tensor)
    mock_output_ids.shape = [len(prompt) + 10]
    mock_llm.generate.return_value = mock_output_ids
    output_ids = mock_llm.generate(prompt, max_new_tokens=10, return_as_token_ids=True)
    assert isinstance(output_ids, torch.Tensor)
    assert output_ids.shape[0] > len(prompt)


def test_calculate_number_of_devices():
    assert calculate_number_of_devices(1) == 1
    assert calculate_number_of_devices([0, 1, 2]) == 3
    assert calculate_number_of_devices(None) == 0


def test_invalid_accelerator(mock_llm):
    with pytest.raises(ValueError, match="Invalid accelerator"):
        LLM.load("path/to/model", accelerator="invalid")


def test_multiple_devices_not_implemented(mock_llm):
    with pytest.raises(NotImplementedError, match="Support for multiple devices is currently not implemented"):
        LLM.load("path/to/model", accelerator="cpu", devices=2)


def test_llm_load_random_init(tmp_path):
    download_from_hub(repo_id="EleutherAI/pythia-14m", tokenizer_only=True, checkpoint_dir=tmp_path)

    torch.manual_seed(123)
    llm = LLM.load(
        model="pythia-160m",
        accelerator="cpu",
        devices=1,
        init="random",
        tokenizer_dir=Path(tmp_path/"EleutherAI/pythia-14m")
    )

    input_text = "some text text"
    output_text = llm.generate(input_text, max_new_tokens=15)
    ln = len(llm.preprocessor.tokenizer.encode(output_text)) - len(llm.preprocessor.tokenizer.encode(input_text))
    assert ln <= 15

    # The following below tests that generate works with different prompt lengths
    # after the kv cache was set

    input_text = "some text"
    output_text = llm.generate(input_text, max_new_tokens=15)
    ln = len(llm.preprocessor.tokenizer.encode(output_text)) - len(llm.preprocessor.tokenizer.encode(input_text))
    assert ln <= 15

    input_text = "some text text text"
    output_text = llm.generate(input_text, max_new_tokens=15)
    ln = len(llm.preprocessor.tokenizer.encode(output_text)) - len(llm.preprocessor.tokenizer.encode(input_text))
    assert ln <= 15

    # Request too big a kvcache size
    with pytest.raises(ValueError):
        output_text = llm.generate(input_text, max_new_tokens=15, max_seq_length=2**63)

    input_text = "Lorem ipsum"
    for max_seq in ('dynamic', 'max_model_supported', 64):
        # Request an amount of tokens that don't fit in the kvcache
        output_text = llm.generate(input_text, max_new_tokens=15, max_seq_length=max_seq)

        # Request too many tokens
        with pytest.raises(ValueError):
            output_text = llm.generate(input_text, max_new_tokens=2**63, max_seq_length=15)


def test_llm_load_hub_init(tmp_path):

    torch.manual_seed(123)
    llm = LLM.load(
        model="EleutherAI/pythia-14m",
        accelerator="cpu",
        devices=1,
        init="pretrained"
    )
    text = llm.generate("text", max_new_tokens=10)
    assert len(text) > 0
