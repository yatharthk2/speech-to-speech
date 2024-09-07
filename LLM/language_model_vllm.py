import os
from threading import Thread
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoConfig
import torch

from LLM.chat import Chat
from baseHandler import BaseHandler
from rich.console import Console
import logging
from nltk import sent_tokenize

logger = logging.getLogger(__name__)
console = Console()

WHISPER_LANGUAGE_TO_LLM_LANGUAGE = {
    "en": "english",
    "fr": "french",
    "es": "spanish",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
}

class LanguageModelHandler(BaseHandler):
    """
    Handles the language model part using vLLM with CUDA debugging options.
    """

    def setup(
        self,
        model_name="neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16",
        device="cuda",
        torch_dtype="float16",
        gen_kwargs={},
        user_role="user",
        chat_size=1,
        init_chat_role=None,
        init_chat_prompt="You are a helpful AI assistant.",
        num_gpus=1,
        cuda_debug=False,
    ):
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)

        # Set CUDA debugging environment variables if requested
        if cuda_debug:
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            os.environ['TORCH_USE_CUDA_DSA'] = '1'

        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your PyTorch installation.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Get the correct max_model_len from the model's config
        config = AutoConfig.from_pretrained(model_name)
        max_model_len = config.max_position_embeddings
        
        # Set the environment variable to allow overriding if needed
        os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

        try:
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=num_gpus,
                max_model_len=max_model_len,
                dtype=self.torch_dtype,
                trust_remote_code=True,
                gpu_memory_utilization=0.8,  # Adjust this value if needed
            )
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise

        self.sampling_params = SamplingParams(
            temperature=gen_kwargs.get('temperature', 0.6),
            top_p=gen_kwargs.get('top_p', 0.9),
            max_tokens=gen_kwargs.get('max_new_tokens', 256),
        )

        self.chat = Chat(chat_size)
        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError(
                    "An initial prompt needs to be specified when setting init_chat_role."
                )
            self.chat.init_chat({"role": init_chat_role, "content": init_chat_prompt})
        self.user_role = user_role

        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        dummy_input_text = "Repeat the word 'home'."
        dummy_chat = [{"role": self.user_role, "content": dummy_input_text}]
        dummy_prompt = self.tokenizer.apply_chat_template(dummy_chat, add_generation_prompt=True, tokenize=False)

        if self.device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()

        try:
            outputs = self.llm.generate(dummy_prompt, self.sampling_params)
        except Exception as e:
            logger.error(f"Error during warmup: {e}")
            raise

        if self.device == "cuda":
            end_event.record()
            torch.cuda.synchronize()

            logger.info(
                f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s"
            )

    def process(self, prompt):
        logger.debug("inferring language model...")
        language_code = None
        if isinstance(prompt, tuple):
            prompt, language_code = prompt
            prompt = f"Please reply to my message in {WHISPER_LANGUAGE_TO_LLM_LANGUAGE[language_code]}. " + prompt

        self.chat.append({"role": self.user_role, "content": prompt})
        full_prompt = self.tokenizer.apply_chat_template(self.chat.to_list(), add_generation_prompt=True, tokenize=False)

        try:
            outputs = self.llm.generate(full_prompt, self.sampling_params)
            generated_text = outputs[0].outputs[0].text
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise

        sentences = sent_tokenize(generated_text)
        for sentence in sentences[:-1]:
            yield (sentence, language_code)

        self.chat.append({"role": "assistant", "content": generated_text})

        # don't forget last sentence
        yield (sentences[-1], language_code)
