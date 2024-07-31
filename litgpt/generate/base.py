# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import sys
import time
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Literal, Optional, List
import warnings

import lightning as L
import torch
import torch._dynamo.config
import torch._inductor.config
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning_utilities.core.imports import RequirementCache

from litgpt.model import GPT
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer
from litgpt.prompts import has_prompt_style, load_prompt_style, PromptStyle
from litgpt.utils import (
    check_file_size_on_cpu_and_warn,
    check_valid_checkpoint_dir,
    extend_checkpoint_dir,
    get_default_supported_precision,
    load_checkpoint
)


def multinomial_num_samples_1(probs: torch.Tensor) -> torch.Tensor:
    if torch._dynamo.is_compiling():
        # Faster alternative to `torch.multinomial(probs, num_samples=1)` that is also CUDAGraph friendly
        distribution = torch.empty_like(probs).exponential_(1)
        return torch.argmax(probs / distribution, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


def sample_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Example:
    # sorted_probs=[0.1, 0.15, 0.2, 0.25, 0.3] -> sorted_cumprobs=[0.1, 0.25, 0.45, 0.7, 1.0]
    # sorted_indices_to_remove = [1, 1, 0, 0, 0] if top_p=0.7
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # Keep at least 1 token always to prevent the case where no token is selected
    # In this case the most probable one is always kept
    sorted_indices_to_remove[-1:] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, float("-inf"))
    return logits


def sample(
    logits: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = None, top_p: float = 1.0
) -> torch.Tensor:
    if top_p < 0.0 or top_p > 1.0:
        raise ValueError(f"top_p must be in [0, 1], got {top_p}")
    logits = logits[0, -1]
    # optionally crop the logits to only the top k options
    if top_k is not None:
        v, i = torch.topk(logits, min(top_k, logits.size(-1)))
        # do not use `torch.where` as in nanogpt because it will repeat top-k collisions
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, i, v)
    # optionally scale the logits and sample from a probability distribution
    if temperature > 0.0 or top_p > 0.0:
        if temperature > 0.0:
            logits = logits / temperature
        # optionally crop the logits to smallest set of logits with a cumulative probability above top_p
        if top_p < 1.0:
            logits = sample_top_p(logits, top_p)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return multinomial_num_samples_1(probs)
    return torch.argmax(logits, dim=-1, keepdim=True)


def next_token(model: GPT, input_pos: torch.Tensor, x: torch.Tensor, debug=False, actions=None, custom_tokens=None, **kwargs: Any) -> torch.Tensor:
    logits = model(x, input_pos)
    if actions != None:
        if len(actions) == 0:
            actions = None
    if debug or actions != None: 
        # Positions to keep
        positions_to_keep = [custom_tokens["switch"], custom_tokens["move"]]
        if actions != None:
            positions_to_keep = actions

        # Create a mask with -inf (negative infinity) for all positions except the ones to keep
        mask = torch.full(logits.shape, float('-inf'), device=logits.device, dtype=logits.dtype)

        # Set the values at the positions to keep to 0
        mask[:, :, positions_to_keep] = 0

        # Apply the mask to the logits
        logits = logits + mask
    next = sample(logits, **kwargs)
    return next.to(dtype=x.dtype)


@torch.inference_mode()
def generate(
    model: GPT,
    prompt: torch.Tensor,
    max_returned_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    eos_id: Optional[int] = None,
    include_prompt: bool = True
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        prompt: Tensor of shape (T) with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return (given plus generated).
        temperature: Scales the predicted logits by 1 / temperature.
        top_k: If specified, only sample among the tokens with the k highest probabilities.
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
        eos_id: If specified, stop generating any more token once the <eos> token is triggered.
        include_prompt: If true (default) prepends the prompt (after applying the prompt style) to the output.
    """
    T = prompt.size(0)
    assert max_returned_tokens > T
    if model.max_seq_length < max_returned_tokens - 1:
        # rolling the kv cache based on the `input_pos` value would be necessary. However, doing so would introduce a
        # data dependency on the `input_pos` tensor and impact model compilation. Since this setting is uncommon, we do
        # not support it to avoid negatively impacting the overall speed
        raise NotImplementedError(f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}")

    device = prompt.device
    if include_prompt:
        tokens = [prompt]
    else:
        tokens = []
    input_pos = torch.tensor([T], device=device)
    token = next_token(
        model, torch.arange(0, T, device=device), prompt.view(1, -1), temperature=temperature, top_k=top_k, top_p=top_p
    ).clone()
    tokens.append(token)
    for _ in range(2, max_returned_tokens - T + 1):
        token = next_token(
            model, input_pos, token.view(1, -1), temperature=temperature, top_k=top_k, top_p=top_p
        ).clone()
        tokens.append(token)
        if token == eos_id:
            break
        input_pos = input_pos.add_(1)
    return torch.cat(tokens)
    

def add_custom_token(custom_token: int, 
                     device, 
                     dtype, 
                     tokens: List[torch.Tensor],
                     model: GPT,
                     input_pos: torch.Tensor,
                     temperature: float,
                     top_k: Optional[int],
                     top_p: float,
                     use_token: bool=True,
                     debug: bool=False,
                     actions: List[int] = None,
                     custom_tokens: List[int] = None
                     ):
    token = torch.tensor([custom_token], device=device, dtype=dtype)
    tokens.append(token)
    input_pos = input_pos.add_(1)
    token_new = next_token(
        model, input_pos, token.view(1, -1), temperature=temperature, top_k=top_k, top_p=top_p, debug=debug, actions=actions, custom_tokens=custom_tokens
    ).clone()
    if use_token:
        tokens.append(token_new)
        input_pos = input_pos.add_(1)
    return tokens

@torch.inference_mode()
def discrete_action_generate(
    model: GPT,
    prompt: torch.Tensor,
    max_returned_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    eos_id: Optional[int] = None,
    include_prompt: bool = True,
    actions: List[List[List[torch.Tensor]]] = None,
    custom_tokens: Dict = None,
    thought_tokens: int = 0
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        prompt: Tensor of shape (T) with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return (given plus generated).
        temperature: Scales the predicted logits by 1 / temperature.
        top_k: If specified, only sample among the tokens with the k highest probabilities.
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
        eos_id: If specified, stop generating any more token once the <eos> token is triggered.
        include_prompt: If true (default) prepends the prompt (after applying the prompt style) to the output.
    """
    T = prompt.size(0)
    assert max_returned_tokens > T
    if model.max_seq_length < max_returned_tokens - 1:
        # rolling the kv cache based on the `input_pos` value would be necessary. However, doing so would introduce a
        # data dependency on the `input_pos` tensor and impact model compilation. Since this setting is uncommon, we do
        # not support it to avoid negatively impacting the overall speed
        raise NotImplementedError(f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}")
    
    device = prompt.device
    if include_prompt:
        tokens = [prompt]
    else:
        tokens = []
    input_pos = torch.tensor([T], device=device)
    # Feed prompt, ignore output
    _ = next_token(
        model, torch.arange(0, T, device=device), prompt.view(1, -1), temperature=temperature, top_k=top_k, top_p=top_p
    ).clone()
    dtype = prompt.view(1, -1)[0].dtype
    # GET THOUGHT
    if thought_tokens != 0:
        # { token, ignore output
        tokens = add_custom_token(5018, device, dtype, tokens, model, input_pos, temperature, top_k, top_p, False, custom_tokens=custom_tokens)
        # add thought id token, ignore output
        tokens = add_custom_token(61665, device, dtype, tokens, model, input_pos, temperature, top_k, top_p, False, custom_tokens=custom_tokens)
        # add ":" token
        tokens = add_custom_token(custom_tokens["\":\""], device, dtype, tokens, model, input_pos, temperature, top_k, top_p, True, custom_tokens=custom_tokens)
        token = tokens[-1]
        # get thought
        assert thought_tokens > 1
        for i in range(1, thought_tokens):
            token = next_token(
                model, input_pos, token.view(1, -1), temperature=temperature, top_k=top_k, top_p=top_p
            ).clone()
            tokens.append(token)
            # if token == eos_id:
            #     break
            input_pos = input_pos.add_(1)
            if token == 498:
                break
            # {"thought":"You "move":"mortalspin"}
            elif i == thought_tokens-1:
                # ", token, ignore output
                tokens = add_custom_token(498, device, dtype, tokens, model, input_pos, temperature, top_k, top_p, False, custom_tokens=custom_tokens)
                break
        # GET DISCRETE ACTION
        # " token
        # PREDICT SWITCH OR MOVE
        tokens = add_custom_token(330, device, dtype, tokens, model, input_pos, temperature, top_k, top_p, True, debug=True, custom_tokens=custom_tokens)
    else:
        # GET DISCRETE ACTION
        # {" token, switch or move output
        # PREDICT SWITCH OR MOVE
        tokens = add_custom_token(custom_tokens["{\""], device, dtype, tokens, model, input_pos, temperature, top_k, top_p, True, debug=True, custom_tokens=custom_tokens)
    # check if action type is valid
    action_token = tokens[-1].item()
    if action_token == custom_tokens["switch"] and len(actions[1]) == 0:
        action_token = custom_tokens["move"]
        tokens[-1] = torch.tensor([action_token], device=device, dtype=tokens[0].dtype)
    if action_token == custom_tokens["move"] and len(actions[0]) == 0:
        action_token = custom_tokens["switch"]
        tokens[-1] = torch.tensor([action_token], device=device, dtype=tokens[0].dtype)

    # Gather valid discrete actions
    valid_tokens = None
    if action_token == custom_tokens["switch"]:   # switch
        valid_tokens = actions[1][:][1:]
        if len(valid_tokens) == 0:
            if len(actions[1]) == 1:
                valid_tokens = actions[1]
    elif action_token == custom_tokens["move"]:  # move
        valid_tokens = actions[0][:][1:]
        if len(valid_tokens) == 0:
            if len(actions[0]) == 1:
                valid_tokens = actions[0]
    else:
        raise ValueError(f'Invalid action token {action_token}\n{tokens}')
    token_counter = 0
    valid_indices = [i for i in range(len(valid_tokens))]
    valid_action_tokens = [valid_tokens[i][token_counter].item() for i in valid_indices]

    # ":" token, token added in for loop
    # tokens = add_custom_token(custom_tokens["\":\""], device, dtype, tokens, model, input_pos, temperature, top_k, top_p, True, actions=valid_action_tokens)
    token = torch.tensor([custom_tokens["\":\""]], device=device, dtype=dtype)
    tokens.append(token)

    output_tokens_remaining = 9
    for _ in range(2, max_returned_tokens + 1):
        token_counter += 1
        token = next_token(
            model, input_pos, token.view(1, -1), temperature=temperature, top_k=top_k, top_p=top_p, actions=valid_action_tokens, custom_tokens=custom_tokens
        ).clone()
        vat_old = valid_action_tokens
        valid_action_tokens = []
        valid_indices_old = valid_indices
        valid_indices = []
        for i, vat in zip(valid_indices_old, vat_old):
            if vat == token.item():
                if len(valid_tokens[i]) > token_counter:
                    valid_action_tokens.append(valid_tokens[i][token_counter].item())
                    valid_indices.append(i)
        tokens.append(token)
        if token == eos_id:
            break
        input_pos = input_pos.add_(1)
        if output_tokens_remaining != 0 and len(valid_action_tokens) != 0:
            output_tokens_remaining -= 1
        else:
            # "} token
            token = torch.tensor([custom_tokens["\"}"]], device=device, dtype=tokens[0].dtype)
            tokens.append(token)
            break
    return torch.cat(tokens)


@torch.inference_mode()
def main(
    checkpoint_dir: Path,
    prompt: str = "What food do llamas eat?",
    *,
    num_samples: int = 1,
    max_new_tokens: int = 50,
    top_k: Optional[int] = 50,
    top_p: float = 1.0,
    temperature: float = 0.8,
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]] = None,
    precision: Optional[str] = None,
    compile: bool = False,
) -> None:
    """Default generation option.

    Generates text samples based on a pre-trained model and tokenizer.

    Args:
        checkpoint_dir: The checkpoint directory to load.
        prompt: The prompt string to use for generating the samples.
        num_samples: The number of text samples to generate.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
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
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            for more details, see https://github.com/Lightning-AI/litgpt/blob/main/tutorials/quantize.md
        precision: Indicates the Fabric precision setting to use.
        compile: Whether to compile the model.
    """
    checkpoint_dir = extend_checkpoint_dir(checkpoint_dir)
    pprint(locals())

    precision = precision or get_default_supported_precision(training=False)

    plugins = None
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

    fabric = L.Fabric(devices=1, precision=precision, plugins=plugins)

    check_valid_checkpoint_dir(checkpoint_dir)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")

    checkpoint_path = checkpoint_dir / "lit_model.pth"
    check_file_size_on_cpu_and_warn(checkpoint_path, fabric.device)

    tokenizer = Tokenizer(checkpoint_dir)
    prompt_style = (
        load_prompt_style(checkpoint_dir) if has_prompt_style(checkpoint_dir) else PromptStyle.from_config(config)
    )

    prompt = prompt_style.apply(prompt)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = GPT(config)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)
    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_returned_tokens
        # enable the kv cache
        model.set_kv_cache(batch_size=1)
    model.eval()

    if compile:
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.coordinate_descent_tuning = True
        global next_token
        next_token = torch.compile(next_token, mode="reduce-overhead")

    model = fabric.setup_module(model)

    t0 = time.perf_counter()
    load_checkpoint(fabric, model, checkpoint_path)
    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    L.seed_everything(1234)
    for i in range(num_samples):
        t0 = time.perf_counter()
        y = generate(model, encoded, max_returned_tokens, temperature=temperature, top_k=top_k, top_p=top_p, eos_id=tokenizer.eos_id)
        t = time.perf_counter() - t0
        for block in model.transformer.h:
            block.attn.kv_cache.reset_parameters()
        fabric.print(tokenizer.decode(y))
        tokens_generated = y.size(0) - prompt_length
        fabric.print(
            f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr
        )
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)
