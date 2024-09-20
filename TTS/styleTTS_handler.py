from styletts2 import tts
import logging
from baseHandler import BaseHandler
import librosa
import numpy as np
from rich.console import Console
import torch

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()


class StyleTTSHandler(BaseHandler):
    def setup(
        self,
        should_listen,
        device="cuda",
        sample_rate=24000,
        diffusion_steps=3,
        stream=True,
        chunk_size=1024,
    ):
        self.should_listen = should_listen
        self.device = device
        self.sample_rate = sample_rate
        self.diffusion_steps = diffusion_steps
        self.stream = stream
        self.chunk_size = chunk_size

        # Initialize the StyleTTS2 model
        try:
            self.model = tts.StyleTTS2()
            logger.info("StyleTTS2 model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize StyleTTS2 model: {e}")
            raise e

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        try:
            _ = self.model.inference(
                "Hello world",
                diffusion_steps=self.diffusion_steps,
                output_sample_rate=self.sample_rate,
            )
            logger.info("Warmup completed successfully.")
        except Exception as e:
            logger.error(f"Warmup failed: {e}")

    def process(self, llm_sentence):
        console.print(f"[green]ASSISTANT: {llm_sentence}")

        if self.device == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()

        try:
            # Perform inference to generate audio data
            audio_data = self.model.inference(
                llm_sentence,
                diffusion_steps=self.diffusion_steps,
                output_sample_rate=self.sample_rate,
            )
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            self.should_listen.set()
            return

        if audio_data is None or len(audio_data) == 0:
            self.should_listen.set()
            return

        # Resample audio if the target sample rate is different
        target_sample_rate = 16000  # Adjust as needed
        if self.sample_rate != target_sample_rate:
            try:
                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=self.sample_rate,
                    target_sr=target_sample_rate,
                )
            except Exception as e:
                logger.error(f"Resampling failed: {e}")
                self.should_listen.set()
                return

        # Scale audio data to int16 format
        try:
            audio_data = (audio_data * 32768).astype(np.int16)
        except Exception as e:
            logger.error(f"Audio data scaling failed: {e}")
            self.should_listen.set()
            return

        # Stream audio data in chunks
        for i in range(0, len(audio_data), self.chunk_size):
            chunk = audio_data[i : i + self.chunk_size]
            if len(chunk) < self.chunk_size:
                chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)))
            yield chunk

        self.should_listen.set()
