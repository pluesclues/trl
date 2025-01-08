# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Usage:

python examples/scripts/dpo_online.py \
    --model_name_or_path keithdrexel/unsloth-llama-3.2-1b-tldr-unsloth-nobnb  \
    --reward_model_path keithdrexel/reward_modeling__meta-llama_Llama-3.2-1B \
    --dataset_name trl-lib/tldr \
    --learning_rate 5.0e-7 \
    --output_dir pythia-1b-tldr-online-dpo \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --warmup_ratio 0.1 \
    --missing_eos_penalty 1.0 \ 
    --logging_steps 20

With LoRA:
python examples/scripts/dpo_online.py \
    --model_name_or_path trl-lib/pythia-1b-deduped-tldr-sft  \
    --reward_model_path trl-lib/pythia-1b-deduped-tldr-rm \
    --dataset_name trl-lib/tldr \
    --learning_rate 5.0e-6 \
    --output_dir pythia-1b-tldr-online-dpo \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --warmup_ratio 0.1 \
    --missing_eos_penalty 1.0 \
    --use_peft
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, GenerationConfig

from trl import (
    HfPairwiseJudge,
    LogCompletionsCallback,
    ModelConfig,
    OnlineDPOConfig,
    OnlineDPOTrainer,
    OpenAIPairwiseJudge,
    PairRMJudge,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from unsloth import FastLanguageModel

JUDGES = {"pair_rm": PairRMJudge, "openai": OpenAIPairwiseJudge, "hf": HfPairwiseJudge}

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, OnlineDPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    # )
    

    if training_args.reward_model_path is not None:
        # reward_model = AutoModelForSequenceClassification.from_pretrained(
        #     training_args.reward_model_path,
        #     num_labels=1,
        #     trust_remote_code=model_args.trust_remote_code,
        #     **model_kwargs,
        # )
        # reward_tokenizer = AutoTokenizer.from_pretrained(
        #     training_args.reward_model_path,
        #     trust_remote_code=model_args.trust_remote_code,
        #     truncation=True,
        #     truncation_side="left",  # since we judge the completion, truncating left is more appropriate
        # )
        FastLanguageModel.reset_functions()
        # create the reward model and optimizer
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            training_args.reward_model_path,
            revision = training_args.reward_model_revision, 
            num_labels=1,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        )
        FastLanguageModel.set_functions()
    else:
        reward_model = None
        reward_tokenizer = None

    if training_args.judge is not None:
        judge_cls = JUDGES[training_args.judge]
        judge = judge_cls()
    else:
        judge = None

    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_args.model_name_or_path,
    #     padding_side="left",
    #     trust_remote_code=model_args.trust_remote_code,
    #     **model_kwargs,
    # )
    
    #Create the models
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
      model_name = model_args.model_name_or_path, # "unsloth/tinyllama" for 16bit loading
      max_seq_length = max_seq_length,
      dtype = dtype,
      load_in_4bit = load_in_4bit,
      token = "", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized 1e-7
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    reward_model_tokenizer  = tokenizer

    tokenizer.padding_side="right"
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    trainer = OnlineDPOTrainer(
        model=model,
        reward_model=reward_model,
        judge=judge,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        reward_processing_class=reward_tokenizer,
        peft_config=get_peft_config(model_args),
    )

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_new_tokens, do_sample=True, temperature=training_args.temperature
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
