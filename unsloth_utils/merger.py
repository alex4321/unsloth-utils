import os
from typing import Dict, List, Tuple, Callable
import safetensors
import safetensors.torch
import torch
import shutil
import json
import pandas as pd
import math


def _load_adapter_weights(adapter_model_directory: str) -> Dict[str, torch.Tensor]:
    adapter_fname = os.path.join(adapter_model_directory, "adapter_model.safetensors")
    assert os.path.exists(adapter_fname), f"Adapter model file {adapter_fname} does not exist"
    prefix = "base_model.model."
    with safetensors.safe_open(adapter_fname, framework="pt", device="cpu") as f:
        result = {}
        for key in f.keys():
            if not key.startswith(prefix):
                continue
            result[key[len(prefix):]] = f.get_tensor(key)
        return result

def _load_adapters_weights(adapter_model_directories: List[str]) -> Tuple[List[Dict[str, torch.Tensor]], List[str]]:
    adapter_weights = [
        _load_adapter_weights(d)
        for d in adapter_model_directories
    ]
    adapter_module_names = []
    for key, _ in adapter_weights[0].items():
        if "lora_A" in key or "lora_B" in key:
            adapter_module = key.split(".lora_")[0]
            if adapter_module not in adapter_module_names:
                adapter_module_names.append(adapter_module)
    return adapter_weights, adapter_module_names

def _get_adapter_delta(adapter_weights: Dict[str, torch.Tensor], module_name: str, lora_scaling: float) -> torch.Tensor:
    with torch.no_grad():
        lora_A = adapter_weights[f"{module_name}.lora_A.weight"]
        lora_B = adapter_weights[f"{module_name}.lora_B.weight"]
        delta = lora_B.matmul(lora_A) * lora_scaling
        return delta

def _get_adapters_delta(adapter_weights: List[Dict[str, torch.Tensor]], module_name: str, lora_scaling: float) -> Dict[str, torch.Tensor]:
    assert len(adapter_weights) > 0, "No adapter weights provided"
    with torch.no_grad():
        delta = _get_adapter_delta(adapter_weights[0], module_name, lora_scaling)
        for i in range(1, len(adapter_weights)):
            delta += _get_adapter_delta(adapter_weights[i], module_name, lora_scaling)
        return delta
    
def _copy_original_configs(source_model_directory: str, output_model_directory: str) -> None:
    if os.path.exists(output_model_directory):
        shutil.rmtree(output_model_directory)
    os.makedirs(output_model_directory, exist_ok=False)
    for item in os.listdir(source_model_directory):
        src_path = os.path.join(source_model_directory, item)
        dst_path = os.path.join(output_model_directory, item)
        if not item.startswith("model") or not item.endswith(".safetensors"):
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

def _load_weight_map(source_model_directory: str) -> pd.DataFrame:
    if "model.safetensors.index" in os.listdir(source_model_directory):
        with open(os.path.join(source_model_directory, 'model.safetensors.index'), 'r') as f:
            weight_map = json.load(f)['weight_map']
            weight_names = sorted(weight_map.keys())
            weight_files = [weight_map[weight_name] for weight_name in weight_names]
            return pd.DataFrame({
                "parameter": weight_names,
                "file": weight_files
            })
    elif "model.safetensors" in os.listdir(source_model_directory):
        with safetensors.safe_open(os.path.join(source_model_directory, 'model.safetensors'), framework="pt", device="cpu") as f:
            weight_names = f.keys()
            weight_files = ["model.safetensors"] * len(weight_names)
            return pd.DataFrame({
                "parameter": weight_names,
                "file": weight_files
            })
    else:
        raise ValueError("Model must either be saved as a single model.safetensors file or must have a model.safetensors.index file.")

def get_lora_scaling(lora_alpha: int, lora_r: int, use_rslora: bool) -> float:
    if use_rslora:
        return lora_alpha / math.sqrt(lora_r)
    else:
        return lora_alpha / lora_r

def _build_adapter_weight_getter(adapter_weights: List[Dict[str, torch.Tensor]], adapter_module_name: List[str], lora_scaling: float) -> Dict[str, Callable[[], torch.Tensor]]:
    parameter = f"{adapter_module_name}.weight"

    def func():
        return _get_adapters_delta(adapter_weights, adapter_module_name, lora_scaling)
    
    return {parameter: func}

def merge(source_model_directory: str, adapter_model_directories: List[str], output_model_directory: str, lora_scaling: float = 1.0) -> bool:
    _copy_original_configs(source_model_directory, output_model_directory)
    index_file_found = 'model.safetensors.index' in os.listdir(source_model_directory)
    safetensors_file_found = 'model.safetensors' in os.listdir(source_model_directory)
    assert index_file_found or safetensors_file_found, "Model must either be saved as a single model.safetensors file or must have a model.safetensors.index file."

    adapter_weights, adapter_module_names = _load_adapters_weights(adapter_model_directories)
    adapter_delta_weight_getters = {}
    for adapter_module_name in adapter_module_names:
        adapter_delta_weight_getters = dict(
            **adapter_delta_weight_getters,
            **_build_adapter_weight_getter(adapter_weights, adapter_module_name, lora_scaling)
        )
    df_weight_map = _load_weight_map(source_model_directory)
    changed = False
    for fname, df_file_weights in df_weight_map.groupby('file'):
        weights = {}
        with torch.no_grad(), \
            safetensors.safe_open(os.path.join(source_model_directory, fname), framework="pt", device="cpu") as f:
            for parameter in df_file_weights['parameter']:
                tensor = f.get_tensor(parameter)
                if parameter in adapter_delta_weight_getters:
                    delta = adapter_delta_weight_getters[parameter]()
                    if not (delta == 0).all():
                        changed = True
                    tensor += delta
                weights[parameter] = tensor
        safetensors.torch.save_file(weights, os.path.join(output_model_directory, fname))
    return changed
