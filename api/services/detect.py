import copy
import json
import math
import os
# os.environ["LOAD_FULL_MODEL"] = "True"
# os.environ["chat_format"] = "chatml"
import torch
import time

from api.model_protector import ModelProtector
from api.utils import LRUCache, get_model_state_dict
from vllm import MLLM, SamplingParams
from api.config import EnvVar
from api.schemas.response import Answer

dectypt = os.getenv('IS_ENCRYPT') != 'false'


class Engine:
    def __init__(self) -> None:
        self.model = LRUCache(1)
        self.base_model = LRUCache(2)

    def load_model(self, model_id, resources_prefix="resources"):
        # 临时判断是否为base model
        load_full_model = EnvVar.LOAD_FULL_MODEL#self._is_base_model(model_id)
        self.resources_prefix = resources_prefix
        self.addapter_resources_prefix = resources_prefix
        if self.model.has(model_id):
            return self.model.get(model_id)
        else:
            if load_full_model:
                # tar -zcf - omchat_v1_7b_n26/ | openssl aes-256-cbc -salt -k HZlh@2023 -out omchat_v1_7b_n26.tar.gz
                if dectypt:
                    status = os.system(
                        "openssl aes-256-cbc -d -salt -k HZlh@2023 -in resources/{}.tar.gz | tar -xz -C resources/".format(
                            model_id))
                    if status != 0:
                        raise RuntimeError("unzip failed, error code is {}. please connect engineer".format(status))
                model = None
                model = MLLM(
                    model="{}/{}".format(resources_prefix, model_id),
                    tokenizer="{}/{}".format(resources_prefix, model_id),
                    dtype="float16",
                    gpu_memory_utilization=EnvVar.GPU_MEMORY_UTILIZATION,
                    max_num_batched_tokens=EnvVar.MAX_NUM_BATCHED_TOKENS,
                    enforce_eager=True
                )
                self.model.put(model_id, model)
                if dectypt:
                    os.system("rm -rf resources/{}".format(model_id))
            else:
                self.model_id = model_id
                if dectypt:
                    a = ModelProtector(xor_key=12, user_id="omchat", model_version=1)
                    encrypt_model_path = os.path.join("{}/{}".format(self.resources_prefix, model_id + '.linker'))
                    if not os.path.exists(encrypt_model_path):
                        error_info = f"{encrypt_model_path} not exists"
                        raise Exception(error_info)
                    try:
                        model_path, out_path = a.decrypt_model(encrypt_model_path)
                        self.addapter_resources_prefix = out_path
                    except:
                        a.remove_model(out_path)
                        raise Exception("解密失败！")

                base_model_id = self._get_base_model_id(model_id)
                base_model = self._load_base_model(base_model_id)
                model = self._load_model_adapter(base_model, model_id)
                if dectypt:
                    a.remove_model(out_path)
            return model


    def _get_base_model_id(self, model_id):
        with open(os.path.join("{}/{}".format(self.addapter_resources_prefix, model_id), "adapter_config.json"),
                  "r") as f:
            adapter_config = json.load(f)
        base_model_id = adapter_config.get("base_model_name_or_path", None)
        if "/" in base_model_id:
            base_model_id = base_model_id.split("/")[-1]
        assert base_model_id is not None, "adapter config has not 'base_model_name_or_path'"
        return base_model_id

    def _load_model_adapter(self, base_model, model_id):
        model = base_model[0].mllm_engine.workers[0].model_runner.model
        base_state_dict = base_model[1]

        model_id_path = os.path.join("{}/{}".format(self.addapter_resources_prefix, model_id))
        model.load_lora_weights(model_id_path, base_state_dict)
        model.to(torch.float16)

        self.model.put(model_id, base_model[0])
        return base_model[0]

    def _load_base_model(self, base_model_id):
        if self.base_model.has(base_model_id):
            return self.base_model.get(base_model_id)
        else:
            if dectypt:
                status = os.system(
                    "openssl aes-256-cbc -d -salt -k HZlh@2023 -in {}/{}.tar.gz | tar -xz -C {}/".format(
                        self.resources_prefix,
                        base_model_id, self.resources_prefix))
                if status != 0:
                    raise RuntimeError("unzip failed, error code is {}. please connect engineer".format(status))

            base_model = MLLM(
                model="{}/{}".format(self.resources_prefix, base_model_id),
                tokenizer="{}/{}".format(self.resources_prefix, base_model_id),
                gpu_memory_utilization=EnvVar.GPU_MEMORY_UTILIZATION,
                dtype="float16",
                lora_weight_id="{}/{}".format(self.addapter_resources_prefix, self.model_id),
                max_num_batched_tokens=EnvVar.MAX_NUM_BATCHED_TOKENS,
                enforce_eager=True
            )
            base_state_dict = {}
            for name, para in get_model_state_dict(base_model).items():
                base_state_dict[name] = copy.deepcopy(para).to("cpu")
            self.base_model.put(base_model_id, (base_model, base_state_dict))
            del base_model
            if dectypt:
                os.system("rm -rf {}/{}".format(self.resources_prefix, base_model_id))
            return self.base_model.get(base_model_id)

    def predict(
            self,
            model_id,
            image,
            src_type,
            text,
            # TODO add choice
            choices,
            initial_prompt,
            temperature=0.2,
            max_tokens=1024,
            top_p=1,
    ):
        model = self.load_model(model_id)
        # TODO: 支持每张图配置不同参数
        # TODO: stop机制
        sampling_params = SamplingParams(
            temperature=temperature, max_tokens=max_tokens, top_p=top_p, stop=["<|im_end|>"]
        )
        if isinstance(text, str):
            text = [{"user": text}]
        res = model.generate(
            prompts=[text],
            images=[{"src_type": src_type, "image_src": image} if image is not None else None],
            sampling_params=sampling_params,
            initial_prompt=initial_prompt,
        )
        generated_text = res[0].outputs[0].text
        input_tokens = len(res[0].prompt_token_ids)
        output_tokens = len(res[0].outputs[0].token_ids)
        return Answer(content=generated_text, input_tokens=input_tokens, output_tokens=output_tokens)

    def batch_predict(
            self,
            model_id,
            prompts,
            initial_prompt,
            temperature=1,
            max_tokens=1024,
            top_p=1,
    ):
        model = self.load_model(model_id)
        sampling_params = SamplingParams(
            temperature=temperature, max_tokens=max_tokens, top_p=top_p, stop=["<|im_end|>"],presence_penalty=1,frequency_penalty=1
        )
        images = []
        texts = []
        choices = []
        for item in prompts:
            images.append({"src_type": item.src_type, "image_src": item.image} if item.image is not None else None)
            texts.append(item.dict()['records'])
            choices.append(item.dict()['choices'])

        res = model.generate(
            prompts=texts,
            images=images,
            choices=choices,
            sampling_params=sampling_params,
            initial_prompt=initial_prompt,
        )
        generated_texts = []
        for output in res:
            text = output.outputs[0].text
            input_tokens = len(output.prompt_token_ids)
            output_tokens = len(output.outputs[0].token_ids)
            mean_prob = None
            probs = {}
            # total_prob = math.exp(output.outputs[0].cumulative_logprob)
            # mean_prob = total_prob**(1/output_tokens)
            # probs={}
            # for name, logprob in output.outputs[0].logprobs:
            #     probs[name] = torch.exp_(torch.tensor(logprob,dtype=torch.float)).item()
            generated_texts.append(Answer(content=text, input_tokens=input_tokens, output_tokens=output_tokens, mean_prob=mean_prob, probs=probs))
            print(output.prompt)
            print(generated_texts)
        return generated_texts


if __name__ == "__main__":
    e = Engine()
    s_t = time.time()
    model = e.load_model(model_id="omchat-llava-qllama-7b-chat-v1-1-finetune_qlora_zh_n67",#"lq_mcqa_0_314",#"omchat-llava-qllama-7b-chat-v1-1-qllama-finetune_zh_n97",#"omchat-llava-vicuna-7b-v1.5-v1-1-finetune_zh_n92",",#
                         resources_prefix="../../../llm_models"
                         )
    print(time.time() - s_t)

    sampling_params = SamplingParams(
        temperature=0.9, max_tokens=512, top_p=1.0, stop=["<|im_end|>"]
    )
    images = []
    texts = []

    res = model.generate(
        prompts=[[{"user": "图片上有什么"}]],
        images=[{"src_type": "url",
                 "image_src": "https://img0.baidu.com/it/u=56109659,3345510515&fm=253&fmt=auto&app=138&f=JPEG?w=889&h=500"}],
        choices=[[]],
        sampling_params=sampling_params,
        initial_prompt="你好",
    )
    generated_texts = []
    for output in res:
        text = output.outputs[0].text
        input_tokens = len(output.prompt_token_ids)
        output_tokens = len(output.outputs[0].token_ids)
        generated_texts.append(Answer(content=text, input_tokens=input_tokens, output_tokens=output_tokens))
        print(output.prompt)
    print(generated_texts)
    print(time.time() - s_t)
    print("done")
