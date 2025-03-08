import os
import pytest
from huggingface_hub import snapshot_download
from unsloth import FastLanguageModel, is_bf16_supported
import torch
import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer
from transformers import TrainingArguments
from joblib import Parallel, delayed
import traceback
from unsloth_utils.merger import merge, get_lora_scaling


@pytest.fixture
def hf_token() -> str:
    assert "HFTOKEN" in os.environ, "HFTOKEN is not set"
    return os.environ["HFTOKEN"]


def get_trainable_model(path: str, lora_alpha: int, lora_r: int):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=path,
        max_seq_length=2048,
        dtype=torch.bfloat16 if is_bf16_supported() else torch.float16,
        load_in_4bit=True,
        load_in_8bit=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=20250308,
        use_rslora=False,
        loftq_config=None,
    )
    return model, tokenizer


def get_dataset(tokenizer: PreTrainedTokenizerFast):
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "test-dataset.csv"))
    df["text"] = df.apply(
        lambda row: tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": row["query"],
                },
                {
                    "role": "assistant",
                    "content": row["response"],
                },
            ],
            tokenize=False,
            add_generation_prompt=False,
        ),
        axis=1
    )
    df = df[["text"]]
    df = pd.concat([df] * 30)
    dataset = Dataset.from_pandas(df)
    return dataset


def train_model(path: str, lora_alpha: int, lora_r: int):
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    model, tokenizer = get_trainable_model(path, lora_alpha, lora_r)
    dataset = get_dataset(tokenizer)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        dataset_num_proc=1,
        max_seq_length=2048,
        packing=False,
        args=TrainingArguments(
            max_grad_norm=1.0,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            eval_accumulation_steps=8,

            warmup_steps=2,
            num_train_epochs=1,
            learning_rate=2e-4,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="constant",
            seed=20250308,
            output_dir=output_dir,
            overwrite_output_dir=True,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_strategy="steps",
            report_to="tensorboard",
        )
    )
    with torch.autograd.detect_anomaly():
        trainer.train()
    return output_dir


def train_model_in_child_process(path: str, lora_alpha: int, lora_r: int):
    def _train_wrapper(path, lora_alpha, lora_r):
        try:
            return train_model(path, lora_alpha, lora_r)
        except Exception:
            return traceback.format_exc()

    result = list(Parallel(n_jobs=1)(
        [delayed(_train_wrapper)(path, lora_alpha, lora_r)]
    ))[0]
    if os.path.exists(result):
        return result
    else:
        raise RuntimeError(f"Training failed in child process:\n{result}")

def latest_checkpoint(path: str):
    checkpoints = []
    for subdir in os.listdir(path):
        if subdir.startswith("checkpoint-"):
            checkpoints.append(os.path.join(path, subdir))
    checkpoints_sorted = sorted(
        checkpoints,
        key=lambda x: int(x.split("checkpoint-")[-1]),
        reverse=True,
    )
    return checkpoints_sorted[0]


def test_merge(hf_token: str):
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    path = snapshot_download(model_name, token=hf_token)
    lora_alpha = 16
    lora_r = 16
    output_dir = train_model_in_child_process(path, lora_alpha, lora_r)
    checkpoint = latest_checkpoint(output_dir)
    merged_dir = os.path.join(os.path.dirname(__file__), "merged")
    is_changed = merge(
        path,
        [checkpoint],
        merged_dir,
        get_lora_scaling(lora_alpha, lora_r, False),
    )
    assert is_changed, "Finetuned model should be changed, but all deltas are 0"
    llm = AutoModelForCausalLM.from_pretrained(merged_dir,
                                               torch_dtype=torch.bfloat16 if is_bf16_supported() else torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(merged_dir)
    input_ids = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Who are you?"
            }
        ],
        return_tensors="pt",
        add_generation_prompt=True,
        tokenize=True,
    )
    with torch.no_grad():
        outputs = llm.generate(input_ids, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assert "unsloth" in response.lower()


