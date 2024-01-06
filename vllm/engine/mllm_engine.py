
import os
import time


from vllm import LLMEngine
from vllm.sampling_params import SamplingParams
from typing import List, Optional
from vllm.logger import init_logger
logger = init_logger(__name__)
from vllm.sequence import Sequence, SequenceGroup


from vllm.engine.conversation import Conversation
import torch
from transformers import PreTrainedTokenizer
from typing import Tuple


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


class MLLMEngine(LLMEngine):

    def tokenizer_image_token(self, prompt, tokenizer, image_token_index=-200, return_tensors=None):
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids

    def make_context(
            self,
            tokenizer: PreTrainedTokenizer,
            query: str,
            choice: List,
            history: List[Tuple[str, str]] = None,
            system: str = "",
            max_window_size: int = 6144, image_token_len=None
    ):
        if history is None:
            history = []

        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [151644]
        im_end_tokens = [151645]
        IMAGE_TOKEN_INDEX = -200
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            """
            if DEFAULT_IMAGE_TOKEN in content:
                return f"{role}\n{content}", tokenizer.encode(
                    role, allowed_special=set()
                ) + nl_tokens + self.tokenizer_image_token(
                    content, tokenizer, IMAGE_TOKEN_INDEX
                )
            else:
            """
            return f"{role}\n{content}", tokenizer.encode(
                role) + nl_tokens + tokenizer.encode(content)

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens
        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            response_text, response_tokens_part = _tokenize_str(
                "assistant", turn_response
            )
            response_tokens = im_start_tokens + response_tokens_part + im_end_tokens
            next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
            prev_chat = (
                f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            )
            current_context_size = (
                    len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text

        if choice:
            choice_postfix = os.getenv("CHOICE_POSTFIX", "请选择正确的答案。")
            query += "\n" + choice_style(choice, choice_postfix)

        context_tokens += (
                nl_tokens
                + im_start_tokens
                + _tokenize_str("user", query)[1]
                + im_end_tokens
                + nl_tokens
                + im_start_tokens
                + tokenizer.encode("assistant")
                + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

        return raw_text, context_tokens

    def add_request(
        self,
        request_id: str,
        prompt: Optional[List[str]],
        image: Optional[dict] = None,
        sampling_params: SamplingParams = None,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        conv: Optional[Conversation] = None,
        choice: Optional[List[str]] = None
    ) -> None:
        """Add a request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            request_id: The unique ID of the request.
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters for text generation.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            arrival_time: The arrival time of the request. If None, we use
                the current time.
        """
        if arrival_time is None:
            arrival_time = time.time()

        image_token_len, mm_use_im_start_end = self._get_image_config()

        prompt, prompt_token_ids = self._get_input_prompt(choice, conv, image, image_token_len, mm_use_im_start_end,
                                                          prompt, prompt_token_ids)

        image_data = image


        # Create the sequences.
        block_size = self.cache_config.block_size
        seqs: List[Sequence] = []
        mcqa_mode = os.getenv("MCAQ_MODE", "only_label")
        choice_token_ids=[]
        if choice:
            if mcqa_mode == "only_label":
                # TODO 选项标签 提取优化
                option_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
                assert len(choice) < len(option_list), "number of choice is > 9, please reduce the number."
                choice_option = option_list[:len(choice)]
                choice_token_ids = self.tokenizer.encode(choice_option)[1:]
                sampling_params.max_tokens = 1
            else:
                max_tokens = max(len(self.tokenizer.encode(s)) for s in choice)
                # choice_token_ids = [[[s] for s in self.tokenizer.encode(s+(max_tokens-len(self.tokenizer.encode(s)))*DEFAULT_IMAGE_PATCH_TOKEN)[1:]] for s in choice]
                # for temp_choices in choice_token_ids:
                #     temp_choice=[]
                #     for padded_choice in choice_token_ids:
                #         temp_choice.append(padded_choice[0][0])
                #     temp_choices.pop(0)
                #     temp_choices.insert(0, temp_choice)
                temp_choice_token_ids = [self.tokenizer.encode(
                    s + (max_tokens + 1 - len(self.tokenizer.encode(s))) * self.tokenizer.special_tokens_map[
                        'eos_token'])[1:] for s in choice]
                for i in range(max_tokens):
                    temp = []
                    for choice_token in temp_choice_token_ids:
                        temp.append(choice_token[i])
                    choice_token_ids.append(temp)

                sampling_params.best_of = len(choice)
                # sampling_params.max_tokens=max_tokens

        for i in range(sampling_params.best_of):
            seq_id = next(self.seq_counter)
            seq = Sequence(seq_id, prompt, prompt_token_ids, block_size, image_data=image_data,
                           choice_token_ids=choice_token_ids)
            seqs.append(seq)

        # Create the sequence group.
        seq_group = SequenceGroup(request_id, seqs, sampling_params,
                                  arrival_time)

        # Add the sequence group to the scheduler.
        self.scheduler.add_seq_group(seq_group)

    def _get_input_prompt(self, choice, conv, image, image_token_len, mm_use_im_start_end, prompt, prompt_token_ids):
        if os.getenv('chat_format') == 'chatml':
            if image is not None:
                prompt, prompt_token_ids = self.make_context(tokenizer=self.tokenizer,
                                                             query=DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + "\n" +
                                                                   prompt[-1]['user'],
                                                             choice=choice,
                                                             history=prompt[:-1],
                                                             system=conv.system if conv != None else '',
                                                             image_token_len=image_token_len)
            else:
                prompt, prompt_token_ids = self.make_context(tokenizer=self.tokenizer,
                                                             query=prompt[-1]['user'],
                                                             choice=choice,
                                                             history=prompt[:-1],
                                                             system=conv.system if conv != None else '')
        else:
            for i, q in enumerate(prompt):
                if choice and i + 1 == len(prompt):
                    choice_postfix = os.getenv("CHOICE_POSTFIX", "请选择正确的答案。")
                    q["user"] += "\n" + choice_style(choice, choice_postfix)

                if i == 0 and image is not None:
                    conv.append_message(conv.roles[0],
                                        self._add_image_token(q["user"], image_token_len, mm_use_im_start_end,
                                                              conv.roles[0]))
                else:
                    conv.append_message(conv.roles[0], q["user"])

                if i + 1 == len(prompt):
                    conv.append_message(conv.roles[1], None)
                else:
                    conv.append_message(conv.roles[1], q["assistant"])

            prompt = conv.get_prompt()
            prompt_token_ids = self.tokenizer.encode(prompt)
        return prompt, prompt_token_ids

    def _get_image_config(self):
        mm_use_im_start_end = self.driver_worker.model_runner.model.model.vision_tower[0].config.use_im_start_end
        image_size = self.driver_worker.model_runner.model.model.vision_tower[0].config.image_size
        patch_size = self.driver_worker.model_runner.model.model.vision_tower[0].config.patch_size
        image_token_len = int((image_size / patch_size) ** 2)
        return image_token_len, mm_use_im_start_end

    def _add_image_token(self, qs, image_token_len, mm_use_im_start_end, role):
        if mm_use_im_start_end:
            qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
        else:
            inp = f"{role}: " + qs
            qs = f"{role}:" + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + '\n' + inp
        return qs


def choice_style(choice: List, postfix: str):
    return "\n" + "选项：" + "\n" + "\n".join(choice) + "\n" + postfix
