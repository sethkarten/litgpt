# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from pathlib import Path
from pprint import pprint
from typing import Dict, Any, Optional
import warnings
from litgpt.utils import check_valid_checkpoint_dir

import lightning as L
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning_utilities.core.imports import RequirementCache
import torch


from litgpt.model import GPT  # needs to be imported before config
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer
from litgpt.generate.base import generate as plain_generate, discrete_action_generate as plain_discrete_action_generate
from litgpt.chat.base import generate as stream_generate
from litgpt.prompts import load_prompt_style, has_prompt_style, PromptStyle
from litgpt.utils import (
    auto_download_checkpoint,
    get_default_supported_precision,
    load_checkpoint
)


_LITSERVE_AVAILABLE = RequirementCache("litserve")
if _LITSERVE_AVAILABLE:
    from litserve import LitAPI, LitServer
else:
    LitAPI, LitServer = object, object


class BaseLitAPI(LitAPI):
    def __init__(self,
                 checkpoint_dir: Path,
                 precision: Optional[str] = None,
                 temperature: float = 0.8,
                 top_k: int = 50,
                 top_p: float = 1.0,
                 max_new_tokens: int = 50) -> None:

        if not _LITSERVE_AVAILABLE:
            raise ImportError(str(_LITSERVE_AVAILABLE))

        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.precision = precision
        self.temperature = temperature
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.discrete_flag = False
        self.thought_tokens = 0

    def setup(self, device: str) -> None:
        # Setup the model so it can be called in `predict`.
        config = Config.from_file(self.checkpoint_dir / "model_config.yaml")
        device = torch.device(device)
        torch.set_float32_matmul_precision("high")

        precision = self.precision or get_default_supported_precision(training=False)
        plugins = None
        quantize = None
        # quantize = "bnb.nf4-dq"
        if quantize is not None and quantize.startswith("bnb."):
            if "mixed" in precision:
                raise ValueError("Quantization and mixed precision is not supported.")
            if RequirementCache("bitsandbytes != 0.42.0"):
                warnings.warn(
                    "LitGPT only supports bitsandbytes v0.42.0. "
                    "This may result in errors when using quantization."
                )
            dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
            plugins = BitsandbytesPrecision(quantize[4:], dtype)
            precision = None


        fabric = L.Fabric(
            accelerator=device.type,
            devices=1 if device.type == "cpu" else [device.index],
            precision=precision,
            plugins=plugins,
        )
        checkpoint_path = self.checkpoint_dir / "lit_model.pth"
        self.tokenizer = Tokenizer(self.checkpoint_dir)
        self.prompt_style = (
            load_prompt_style(self.checkpoint_dir)
            if has_prompt_style(self.checkpoint_dir)
            else PromptStyle.from_config(config)
        )
        with fabric.init_module(empty_init=True):
            model = GPT(config)

        # This should be set if we add a compile feature later
        # with fabric.init_tensor():
        #     model.set_kv_cache(batch_size=1)

        model.eval()

        self.model = fabric.setup_module(model)
        load_checkpoint(fabric, self.model, checkpoint_path)
        self.device = fabric.device

        # get custom tokens for discrete actions
        custom_tokens = ["{\"","switch", "move", "\":\"", "\"}"]
        self.custom_tokens = {}
        for token in custom_tokens:
            self.custom_tokens[token] = self.tokenizer.token_to_id(token) 
        print(self.custom_tokens)

    def decode_request(self, request: Dict[str, Any]) -> Any:
        # Convert the request payload to your model input.
        system_prompt = ""
        if "system" in request.keys():
            system_prompt = request["system"]
        else:
            print('no system prompt found')
        if "temperature" in request.keys():
            self.temperature = float(request["temperature"])
        if "max_tokens" in request.keys():
            self.max_new_tokens = int(request["max_tokens"])
        if "discrete" in request.keys():
            self.discrete_flag = bool(request["discrete"])
        if "thought_tokens" in request.keys():
            self.thought_tokens = int(request["thought_tokens"])
        prompt = request["prompt"]
        # print(request['actions'])
        encoded_actions = None
        if "actions" in request.keys():
            encoded_moves = []
            for move in request['actions'][0]:
                encoded_moves.append(self.tokenizer.encode(move, device=self.device))
            encoded_switches = []
            for switch in request['actions'][1]:
                encoded_switches.append(self.tokenizer.encode(switch, device=self.device))
            encoded_actions = [encoded_moves, encoded_switches]
        prompt = self.prompt_style.apply(prompt, system_prompt=system_prompt)
        encoded = self.tokenizer.encode(prompt, device=self.device)
        return [encoded, encoded_actions]


class SimpleLitAPI(BaseLitAPI):
    def __init__(self,
                 checkpoint_dir: Path,
                 precision: Optional[str] = None,
                 temperature: float = 0.8,
                 top_k: int = 50,
                 top_p: float = 1.0,
                 max_new_tokens: int = 50):
        super().__init__(checkpoint_dir, precision, temperature, top_k, top_p, max_new_tokens)   

    def setup(self, device: str):
        super().setup(device)

    def predict(self, inputs: torch.Tensor) -> Any:
        # Run the model on the input and return the output.
        inputs, actions = inputs[0], inputs[1]
        prompt_length = inputs.size(0)
        max_returned_tokens = prompt_length + self.max_new_tokens
        first_turn = self.model.mask_cache is None
        if first_turn or max_returned_tokens > self.model.max_seq_length:
            self.model.max_seq_length = max_returned_tokens
            self.model.set_kv_cache(batch_size=1, device=self.device)
            
        # generate from set of discrete actions
        if self.discrete_flag:
            y = plain_discrete_action_generate(
                self.model,
                inputs,
                max_returned_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                eos_id=self.tokenizer.eos_id,
                include_prompt=False,
                actions=actions,
                custom_tokens=self.custom_tokens,
                thought_tokens=self.thought_tokens
            )
        else:
            y = plain_generate(
                self.model,
                inputs,
                max_returned_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                eos_id=self.tokenizer.eos_id,
                include_prompt=False
            )

        self.model.clear_kv_cache()
        return y

    def encode_response(self, output: torch.Tensor) -> Dict[str, Any]:
        # Convert the model output to a response payload.
        decoded_output = self.tokenizer.decode(output)
        return {"output": decoded_output}


class StreamLitAPI(BaseLitAPI):
    def __init__(self,
                 checkpoint_dir: Path,
                 precision: Optional[str] = None,
                 temperature: float = 0.8,
                 top_k: int = 50,
                 top_p: float = 1.0,
                 max_new_tokens: int = 50):
        super().__init__(checkpoint_dir, precision, temperature, top_k, top_p, max_new_tokens)   

    def setup(self, device: str):
        super().setup(device)

    def predict(self, inputs: torch.Tensor) -> Any:
        # Run the model on the input and return the output.
        prompt_length = inputs.size(0)
        max_returned_tokens = prompt_length + self.max_new_tokens

        first_turn = self.model.mask_cache is None
        if first_turn or max_returned_tokens > self.model.max_seq_length:
            self.model.max_seq_length = max_returned_tokens
            self.model.set_kv_cache(batch_size=1, device=self.device)

        try:
            yield from stream_generate(
                self.model,
                inputs,
                max_returned_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                stop_tokens=([self.tokenizer.eos_id],)
            )
        finally:
            self.model.clear_kv_cache()

    def encode_response(self, output):
        for out in output:
            yield {"output": self.tokenizer.decode(out)}


def run_server(
    checkpoint_dir: Path,
    precision: Optional[str] = None,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 1.0,
    max_new_tokens: int = 50,
    devices: int = 1,
    accelerator: str = "auto",
    port: int = 8000,
    stream: bool = False,
    access_token: Optional[str] = None,
) -> None:
    """Serve a LitGPT model using LitServe.

    Evaluate a model with the LM Evaluation Harness.

    Arguments:
        checkpoint_dir: The checkpoint directory to load the model from.
        precision: Optional precision setting to instantiate the model weights in. By default, this will
            automatically be inferred from the metadata in the given ``checkpoint_dir`` directory.
        temperature: Temperature setting for the text generation. Value above 1 increase randomness.
            Values below 1 decrease randomness.
        top_k: The size of the pool of potential next tokens. Values larger than 1 result in more novel
            generated text but can also lead to more incoherent texts.
        top_p: If specified, it represents the cumulative probability threshold to consider in the sampling process.
            In top-p sampling, the next token is sampled from the highest probability tokens
            whose cumulative probability exceeds the threshold `top_p`. When specified,
            it must be `0 <= top_p <= 1`. Here, `top_p=0` is equivalent
            to sampling the most probable token, while `top_p=1` samples from the whole distribution.
            It can be used in conjunction with `top_k` and `temperature` with the following order
            of application:

            1. `top_k` sampling
            2. `temperature` scaling
            3. `top_p` sampling

            For more details, see https://arxiv.org/abs/1904.09751
            or https://huyenchip.com/2024/01/16/sampling.html#top_p
        max_new_tokens: The number of generation steps to take.
        devices: How many devices/GPUs to use.
        accelerator: The type of accelerator to use. For example, "auto", "cuda", "cpu", or "mps".
            The "auto" setting (default) chooses a GPU if available, and otherwise uses a CPU.
        port: The network port number on which the model is configured to be served.
        stream: Whether to stream the responses.
        access_token: Optional API token to access models with restrictions.
    """
    checkpoint_dir = auto_download_checkpoint(model_name=checkpoint_dir, access_token=access_token)
    pprint(locals())

    if not stream:
        server = LitServer(
            SimpleLitAPI(
                checkpoint_dir=checkpoint_dir,
                precision=precision,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                ),
            accelerator=accelerator,
            devices=devices,
            timeout=100
            )

    else:
        server = LitServer(
            StreamLitAPI(
                checkpoint_dir=checkpoint_dir,
                precision=precision,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                ),
            accelerator=accelerator,
            devices=devices,
            stream=True
            )

    server.run(port=port, generate_client_file=False)
