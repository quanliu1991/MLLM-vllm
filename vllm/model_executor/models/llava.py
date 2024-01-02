# coding=utf-8
# Adapted from
# https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava.py
# Copyright 2023 The Omlab team.
# Copyright 2022 Haotian Liu and vLLM and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on Haotian Liu's llava.
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
import json
import os
import re
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import LlamaConfig, CLIPVisionModel, CLIPImageProcessor, AutoConfig, \
    AutoModelForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from vllm.config import ModelConfig
from vllm.model_executor import InputMetadata
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.models.llama import LlamaModel, KVCache
from vllm.model_executor.parallel_utils.parallel_state import get_tensor_model_parallel_rank
from vllm.model_executor.layers.linear import ColumnParallelLinear, LinearMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput

# from vllm.model_executor.weight_utils import hf_model_weights_iterator, load_tensor_parallel_weights

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
# IMAGE_PATCH_ID = 32000

CLIP_MODEL_MAP = {}


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(nn.Module):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig, mm_vision_tower=None, mm_hidden_size=None):
        super().__init__()
        self.llama_model = LlamaModel(config)
        if hasattr(config, "mm_vision_tower"):
            # HACK: for FSDP
            mm_vision_tower = config.mm_vision_tower.split("/")[-1]
            clip_model = os.path.abspath(config._name_or_path) + "/" + mm_vision_tower
            assert os.path.isdir(
                clip_model), f"not find {clip_model} dir. please check 'config.json' mm_vision_tower model in {config._name_or_path} "
            self.vision_tower = [CLIPVisionModel.from_pretrained(clip_model, torch_dtype=torch.float16).cuda()]
            self.image_processor = CLIPImageProcessor.from_pretrained(clip_model, torch_dtype=torch.float16)
            # self.image_processor = CLIPImageProcessor.from_pretrained(self.config_class.mm_vision_tower,torch_dtype=torch.float16)

        if hasattr(config, "use_mm_proj"):
            projector_type = getattr(config, 'mm_projector_type', 'linear')
            if projector_type == 'linear':
                self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)

            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU())
                    modules.append(nn.Linear(config.hidden_size, config.hidden_size))
                self.mm_projector = nn.Sequential(*modules)

    def initialize_vision_modules(self, vision_tower, mm_vision_select_layer,
                                  pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False):
        self.config.mm_vision_tower = vision_tower

        image_processor = CLIPImageProcessor.from_pretrained(self.config_class.mm_vision_tower,
                                                             torch_dtype=torch.float16)

        if not hasattr(self, 'vision_tower'):
            vision_tower = CLIPVisionModel.from_pretrained(vision_tower)
        else:
            vision_tower = self.vision_tower[0]
        vision_tower.requires_grad_(False)
        vision_tower = vision_tower.to(torch.float16)
        self.vision_tower = [vision_tower]

        vision_config = vision_tower.config
        num_patches = (vision_config.image_size // vision_config.patch_size) ** 2

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_config.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer

        if not hasattr(self, 'mm_projector'):
            self.mm_projector = nn.Linear(vision_config.hidden_size, self.config.hidden_size)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            self.mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        return dict(
            image_processor=image_processor,
            image_token_len=num_patches,
            vision_config=vision_config
        )

    def forward(self,
                input_ids: None,
                image_datas: List,
                positions: torch.Tensor,
                kv_caches: List[KVCache],
                input_metadata: InputMetadata,
                cache_events: Optional[List[torch.cuda.Event]],

                ) -> Union[Tuple, BaseModelOutputWithPast]:
        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)

        inputs_embeds = None

        batch_images = image_datas
        if image_datas is not None:
            for i in range(len(image_datas)):
                if image_datas[i] is None:
                    batch_images = None
                    break

        def convert(tensor, old_value, new_value):
            tensor[tensor == old_value] = new_value
            return tensor

        def contains_value(tensor, value):
            return torch.any(tensor == value).item()

        # if os.getenv('chat_format') == 'chatml':
        #     input_ids = convert(input_ids.clone(), 151851, IMAGE_PATCH_ID)
        # else:
        #     input_ids = convert(input_ids.clone(), 32000, IMAGE_PATCH_ID)

        if inputs_embeds is None:
            inputs_embeds = self.llama_model.embed_tokens(input_ids)

        vision_tower = getattr(self, 'vision_tower', None)

        def _is_have_image(batch_images):
            if batch_images is None:
                return False
            for image in batch_images:
                if len(image) != 0:
                    return True
            return False

        if vision_tower is not None and _is_have_image(batch_images):
            # TODO: this is a modified multimodal LLM -- Haotian Liu
            vision_tower = vision_tower[0]  # HACK: for FSDP
            with torch.no_grad():
                batch_image_tensors = []
                for images in batch_images:
                    if images is None:
                        batch_image_tensors.append(images)
                    elif type(images) is list:
                        # variable length images
                        image_features = []

                        for image in images:
                            image_forward_out = vision_tower(image.unsqueeze(0), output_hidden_states=True)
                            # image_forward_out = vision_tower(image, output_hidden_states=True)
                            select_hidden_state_layer = getattr(self.llama_model.config, "mm_vision_select_layer", -1)
                            select_hidden_state = image_forward_out.hidden_states[select_hidden_state_layer]
                            image_feature = select_hidden_state[:, 1:]
                            image_feature = self.mm_projector(image_feature)[0]
                            image_features.append(image_feature)
                        batch_image_tensors.append(image_features)
                    else:
                        image_forward_outs = vision_tower(images.unsqueeze(0), output_hidden_states=True)
                        select_hidden_state_layer = getattr(self.llama_model.config, "mm_vision_select_layer", -1)
                        select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
                        image_features = select_hidden_state[:, 1:]
                        image_features = self.mm_projector(image_features)[0]
                        batch_image_tensors.append([image_features])

            # updata input embed
            inputs_embeds = self.updata_input_embed(input_ids, inputs_embeds, batch_image_tensors, vision_tower)
        else:
            inputs_embeds = self.llama_model.embed_tokens(input_ids)

        hidden_states = self.llama_model(input_ids, positions, kv_caches,
                                         input_metadata, inputs_embeds=inputs_embeds)
        return hidden_states

    def updata_input_embed(self, input_ids, inputs_embeds, batch_image_tensors, vision_tower):
        image_tensors_list = [image_tensors[0] for image_tensors in batch_image_tensors]
        use_im_start_end = vision_tower.config.use_im_start_end
        if use_im_start_end:
            image_start_id = vision_tower.config.im_start_token
            image_end_id = vision_tower.config.im_end_token
        image_patch_id = vision_tower.config.im_patch_token
        if image_tensors_list:
            number_patch = int(image_tensors_list[0].shape[0])
        else:
            return inputs_embeds

        def get_image_embed_index(input_ids, use_im_start_end):
            img_embed_start_indexs = []
            img_embed_end_indexs = []
            for i in range(len(input_ids)):
                if use_im_start_end:
                    img_embed_start_indexs.append(
                        [_id.item() + 1 for _id in torch.where(input_ids == image_start_id)[0]])
                    img_embed_end_indexs.append([_id.item() for _id in torch.where(image_end_id == input_ids)[0]])
                else:
                    img_patch_indexs = [_id.item() for _id in torch.where(input_ids[i] == image_patch_id)[0]]
                    img_embed_start_indexs.append(img_patch_indexs[0])
                    img_embed_end_indexs.append(img_patch_indexs[number_patch - 1])
            return img_embed_start_indexs, img_embed_end_indexs

        img_embed_start_indexs, img_embed_end_indexs = get_image_embed_index(input_ids, use_im_start_end)
        for i in range(input_ids.shape[0]):
            inputs_embeds[i][img_embed_start_indexs[i]:img_embed_end_indexs[i] + 1, :] = \
                batch_image_tensors[i][0]

        return inputs_embeds

    def get_batch_inputs(self, input_ids, inputs_embeds, positions):
        device = input_ids.device
        input_ids_list = list(input_ids.cpu().numpy())
        inputs_embeds_list = list(inputs_embeds.cpu().numpy())
        batch_input_ids, batch_inputs_embeds = [], []
        temp_ids, temp_embed = [], []
        for input_id, inputs_embed, position in zip(input_ids_list, inputs_embeds_list, positions):
            if position == 0:
                if temp_ids and temp_embed:
                    batch_input_ids.append(torch.tensor(temp_ids, device=device))
                    batch_inputs_embeds.append(torch.tensor(temp_embed, device=device))
                temp_ids, temp_embed = [], []
            temp_ids.append(input_id)
            temp_embed.append(inputs_embed)
        if temp_ids and temp_embed:
            batch_input_ids.append(torch.tensor(temp_ids, device=device))
            batch_inputs_embeds.append(torch.tensor(temp_embed, device=device))
        return batch_input_ids, batch_inputs_embeds


class LlavaLlamaForCausalLM(nn.Module):
    config_class = LlavaConfig

    def __init__(self, config, linear_method: Optional[LinearMethodBase] = None):
        super().__init__()
        if isinstance(config, ModelConfig):
            config.hf_config._name_or_path = config.base_model
            self.config = config = config.hf_config
        else:
            self.config = config

        self.model = LlavaLlamaModel(config)

        self.lm_head = ColumnParallelLinear(config.hidden_size,
                                            config.vocab_size,
                                            bias=False,
                                            gather_output=False, )
        self.sampler = Sampler(config.vocab_size)

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.Tensor,
            image_datas: List,
            positions: torch.Tensor,
            kv_caches: List[KVCache],
            input_metadata: InputMetadata,
            cache_events: Optional[List[torch.cuda.Event]],
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        hidden_states = self.model(input_ids, image_datas, positions, kv_caches,
                                   input_metadata, cache_events)
        # next_tokens = self.sampler(self.lm_head.weight, hidden_states,
        #                            input_metadata)
        return hidden_states

    def sample(
            self,
            hidden_states: torch.Tensor,
            sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                   sampling_metadata)
        return next_tokens

    _column_parallel_weights = [
        "embed_tokens.weight", "lm_head.weight", "qkv_proj.weight",
        "gate_proj.weight", "up_proj.weight"
    ]

    _row_parallel_weights = ["o_proj.weight", "down_proj.weight"]

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):

            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if "vision_tower" in name:
                continue

            if "mm_projector" in name:
                pass
            else:
                name = name.replace("model", "model.llama_model")

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

    def load_lora_weights(self,
                          model_name_or_path: str,
                          base_state_dict: dict,
                          cache_dir: Optional[str] = None,
                          load_format: str = "auto",
                          revision: Optional[str] = None):

        if model_name_or_path is None:
            state_dict = dict(self.named_parameters())
            device=None
            for name, weight in base_state_dict.items():
                device = state_dict[name].device if device is None else device
                state_dict[name].data.copy_(weight.to(device))
            return

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        state_dict = dict(self.named_parameters())
        self.lora_weight_dict = {}
        self.lora_weight_name = []
        with open(os.path.join(model_name_or_path, "adapter_config.json"), "r") as config_file:
            get_lora_config = json.load(config_file)
        self.scaling = get_lora_config.get("lora_alpha") / get_lora_config.get("r")
        self.target_modules = get_lora_config.get("target_modules")
        self.fan_in_fan_out = get_lora_config.get("fan_in_fan_out")

        for lora_name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            loaded_weight = loaded_weight.to(torch.float16)
            if "mm_projector" in lora_name:
                profix_name = lora_name.split(".mm_projector.")[0]
                lora_name = lora_name.replace(profix_name, "model")
                if "lora_" in lora_name:
                    self.lora_weight_dict[lora_name] = loaded_weight
                    name = re.sub(r'lora_*..', "", lora_name)
                    if self._is_lora_name_match(name):
                        loaded_weight = self._get_delta_weight(name)
                    else:
                        self.lora_weight_name.append(name)
                        continue
                else:
                    name = lora_name

            elif "lm_head" in lora_name:
                profix_name = lora_name.split(".lm_head.")[0]
                name = lora_name.replace(profix_name, "")

            elif "embed_tokens" in lora_name:
                profix_name = lora_name.split(".embed_tokens.")[0]
                name = lora_name.replace(profix_name, "model.llama_model")



            elif "layers" in lora_name:
                profix_name = lora_name.split(".layers.")[0]
                lora_name = lora_name.replace(profix_name, "model.llama_model")
                self.lora_weight_dict[lora_name] = loaded_weight
                name = re.sub(r'lora_*..', "", lora_name)
                if self._is_lora_name_match(name):
                    loaded_weight = self._get_delta_weight(name)
                else:
                    self.lora_weight_name.append(name)
                    continue
            else:
                name = lora_name

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in state_dict:
                    continue

                base_param = base_state_dict[name]
                param = state_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id, base_param=base_param)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in state_dict:
                    continue
                base_param = base_state_dict[name]
                param = state_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight, base_param=base_param)

    def _is_lora_name_match(self, lora_name):
        if lora_name in self.lora_weight_name:
            return True
        else:
            return False

    def _get_delta_weight(self, name) -> torch.Tensor:
        lora_A_name = name.replace("weight", "lora_A.weight")
        lora_B_name = name.replace("weight", "lora_B.weight")
        return (
                transpose(
                    self.lora_weight_dict[lora_B_name].to(self.lm_head.weight.device) @ self.lora_weight_dict[
                        lora_A_name].to(self.lm_head.weight.device),
                    self.fan_in_fan_out
                )
                * self.scaling
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    def initialize_vision_tokenizer(self, tokenizer):
        old_embeddings = self.model.llama_model.embed_tokens
        mm_use_im_start_end = getattr(self.model.llama_model.config, "mm_use_im_start_end", None)
        assert mm_use_im_start_end is not None, "please 'use_im_start_end' in llama config."
        # device
        vision_config = self.get_model().vision_tower[0].config
        vision_config.use_im_start_end = mm_use_im_start_end
        tokenizer.add_special_tokens({"pad_token": DEFAULT_IMAGE_PATCH_TOKEN})
        self.model.llama_model.embed_tokens = self.resize_token_embeddings(old_embeddings, len(tokenizer))
        # tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)

        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
            self.model.llama_model.embed_tokens = self.resize_token_embeddings(old_embeddings, len(tokenizer))
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]

    def resize_token_embeddings(self, old_embeddings, new_num_tokens):
        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        if old_num_tokens == new_num_tokens:
            return old_embeddings
        # Build new embeddings
        new_embeddings = VocabParallelEmbedding(
            new_num_tokens,
            old_embedding_dim)
        new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)

        # initialize all new embeddings (in particular added tokens)
        # self._init_weights(new_embeddings)
        n = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
        return new_embeddings


def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight


# AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
