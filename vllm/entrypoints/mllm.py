import os
import time
import base64
from io import BytesIO
import aiohttp
from linker_atom.lib.load_image import mmap_to_pil
from PIL import Image
from typing import List, Optional, Union
import asyncio
from tqdm import tqdm

from vllm import LLM, EngineArgs
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.worker.model_runner import MModelRunner, CUDAMGraphRunner
from vllm.engine.conversation import (Conversation, SeparatorStyle,
                                      conv_templates)
from vllm.engine.mllm_engine import MLLMEngine
from vllm.utils import Counter
from vllm.logger import init_logger

logger = init_logger(__name__)


class MLLM(LLM):
    def __init__(
            self,
            model: str,
            tokenizer: Optional[str] = None,
            tokenizer_mode: str = "auto",
            trust_remote_code: bool = False,
            tensor_parallel_size: int = 1,
            dtype: str = "auto",
            quantization: Optional[str] = None,
            revision: Optional[str] = None,
            tokenizer_revision: Optional[str] = None,
            seed: int = 0,
            gpu_memory_utilization: float = 0.9,
            swap_space: int = 4,
            enforce_eager: bool = False,
            max_context_len_to_capture: int = 8192,
            **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        if "lora_weight_id" in kwargs.keys():
            kwargs["base_model_id"] = model
            model = kwargs.pop("lora_weight_id")

        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            model_runner=MModelRunner,
            cuda_graph_runner=CUDAMGraphRunner,
            **kwargs,
        )
        self.mllm_engine = MLLMEngine.from_engine_args(engine_args)
        self.request_counter = Counter()

        #TODO standardization chat_format
        if os.getenv('chat_format',None) is not None:
            self.conv_mode = os.getenv('chat_format')
        else:
            self.conv_mode = 'v1'


    async def generate(
            self,
            prompts: Optional[Union[str, List[str]]] = None,
            images: Optional[Union[dict, List[dict]]] = None,
            choices: Optional[List[List[str]]] = None,
            sampling_params: Optional[SamplingParams] = None,
            prompt_token_ids: Optional[List[List[int]]] = None,
            use_tqdm: bool = False,
            initial_prompt: Optional[str] = None,
    ) -> List[RequestOutput]:
        """Generates the completions for the input prompts.

        NOTE: This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: A list of prompts to generate completions for.
            images: A list of images.
            choices: A list of choices.
            sampling_params: The sampling parameters for text generation. If
                None, we use the default sampling parameters.
            prompt_token_ids: A list of token IDs for the prompts. If None, we
                use the tokenizer to convert the prompts to token IDs.
            use_tqdm: Whether to use tqdm to display the progress bar.
            initial_prompt: The prompt that will be use in the gen by default.

        Returns:
            A list of `RequestOutput` objects containing the generated
            completions in the same order as the input prompts.
        """
        st = time.time()
        if prompts is None and prompt_token_ids is None:
            raise ValueError("Either prompts or prompt_token_ids must be " "provided.")

        if isinstance(prompts, str):
            # Convert a single prompt to a list.
            prompts = [prompts]

        assert len(prompts) == len(images), (
            f"The number of images entered should be the same as the number of text，get image number is "
            f"{len(images)} but text number is {len(prompts)}. "
            "if image is None, please use {} placeholder."
        )

        assert len(prompts) == len(choices), (
            f"The number of choices entered should be the same as the number of text，get choice number is "
            f"{len(choices)} but text number is {len(prompts)}."
            "if choice is None, please use [] placeholder."
        )


        if (prompts is not None and prompt_token_ids is not None
                and len(prompts) != len(prompt_token_ids)):
            raise ValueError("The lengths of prompts and prompt_token_ids "
                             "must be the same.")
        if sampling_params is None:
            # Use default sampling params.
            sampling_params = SamplingParams()


        # TODO To support multiple conv_templates
        if self.conv_mode in conv_templates:
            conv_template = conv_templates[self.conv_mode]
        else:
            conv_template= conv_templates["vicuna_v1"]

        if initial_prompt:
            conv_template.system=initial_prompt


        # Add requests to the engine.
        num_requests = len(prompts) if prompts is not None else len(
            prompt_token_ids)

        image_datas = await self.load_image(images)

        for i in range(num_requests):
            conv = conv_template.copy()
            prompt = prompts[i] if prompts is not None else None
            token_ids = prompt_token_ids[i] if prompt_token_ids is not None else None
            image = image_datas[i]
            choice = choices[i]
            self._add_request(prompt, sampling_params, token_ids, image, conv, choice)
        et = time.time()
        logger.warning("_add_request:{}".format(et - st))
        st = time.time()
        result = self._run_engine(use_tqdm)
        et = time.time()
        logger.warning("self._run_engine(use_tqdm):{}".format(et - st))
        # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # for item in result:
        #     for out in item.outputs:
        #         if out.text.endswith(stop_str):
        #             out.text = out.text[: -len(stop_str)]
        return result

    def image_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            return encoded_string.decode('utf-8')

    async def _load_one_image(self, image_src):
        # time.sleep(5)
        # print("_load_one_image")
        if not image_src:
            return None
        image_file = image_src.get("image_src")
        src_type = image_src.get("src_type")

        if src_type == "url":
            async with aiohttp.ClientSession() as session:
                async with session.get(image_file, timeout=30) as response:
                    image_file = await response.read()
            image = Image.open(BytesIO(image_file)).convert('RGB')
        elif src_type == "local":
            image = Image.open(image_file).convert('RGB')
        elif src_type == "base64":
            image = Image.open(BytesIO(base64.b64decode(image_file))).convert('RGB')
        elif src_type == "mmap":
            image = mmap_to_pil(value=image_file)
        else:
            assert 0, "src_type is not true"

        image_tensor = \
            self.mllm_engine.driver_worker.model_runner.model.model.image_processor(image, return_tensors='pt')[
                'pixel_values'][0]
        return image_tensor.half().cuda()

    async def load_image(self, image_srcs):

        image_srcs = image_srcs if isinstance(image_srcs, list) else [image_srcs]
        st = time.time()
        tasks = [asyncio.create_task(self._load_one_image(image_src)) for image_src in image_srcs]
        images = await asyncio.gather(*tasks)
        et = time.time()
        logger.warning("parallel_backend:{}".format(et - st))
        return images

    def _add_request(
            self,
            prompt: Optional[List[str]],
            sampling_params: SamplingParams,
            prompt_token_ids: Optional[List[int]],
            image: Optional[dict] = None,
            conv: Optional[Conversation] = None,
            choice: Optional[List[str]] = None
    ) -> None:
        request_id = str(next(self.request_counter))

        self.mllm_engine.add_request(
            request_id=request_id,
            prompt=prompt,
            image=image,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
            conv=conv,
            choice=choice
        )

    def _run_engine(self, use_tqdm: bool) -> List[RequestOutput]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.mllm_engine.get_num_unfinished_requests()
            pbar = tqdm(total=num_requests, desc="Processed prompts")
        # Run the engine.
        outputs: List[RequestOutput] = []
        prompt_time = None
        decode_time = []
        while self.mllm_engine.has_unfinished_requests():
            st = time.time()
            step_outputs = self.mllm_engine.step()
            et = time.time()
            if prompt_time is None:
                prompt_time = et - st
            else:
                decode_time.append(et-st)
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    if use_tqdm:
                        pbar.update(1)

        logger.warning("prompt_time:{}".format(prompt_time))
        if len(decode_time)>0:
            logger.warning("decode_time:{}".format(sum(decode_time)/len(decode_time)))

        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        return outputs


if __name__ == "__main__":
    model = "/app/vllm/llm_models/qllama-7b-chat_bk"
    # lora_weight_id = "/app/vllm/llm_models/omchat_mcqa_1_2519"
    lora_weight_id = "/app/vllm/llm_models/omchat-llava-qllama-7b-chat-v1-1-finetune_qlora_zh_n67"

    model = MLLM(model=model, tokenizer=model, lora_weight_id=lora_weight_id, dtype="float16",
                 max_num_batched_tokens=6144)
    sampling_params = SamplingParams(
        temperature=0.9, max_tokens=512, top_p=1, stop=["<|im_end|>"], presence_penalty=1,
        frequency_penalty=1
    )
    images = [{"src_type": "url",
               "image_src": "https://img0.baidu.com/it/u=56109659,3345510515&fm=253&fmt=auto&app=138&f=JPEG?w=889&h=500"}
              for _ in range(10)]
    prompts = [[{"user": "你是谁"}] for _ in range(10)]
    choices = [["A", "B", "C", "D"] for _ in range(10)]

    res = model.generate(
        prompts=prompts,
        images=images,
        choices=choices,
        sampling_params=sampling_params,
        initial_prompt="你好",
    )
    print(res)
