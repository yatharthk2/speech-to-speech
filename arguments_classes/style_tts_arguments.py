from dataclasses import dataclass, field

@dataclass
class StyleTTSHandlerArguments:
    style_tts_stream: bool = field(
        default=True,
        metadata={"help": "Enable streaming mode for StyleTTS. Default is True."},
    )
    style_tts_device: str = field(
        default="cuda",
        metadata={
            "help": "The device to be used for StyleTTS speech synthesis. Default is 'cuda'. Options are 'cuda', 'cpu', or 'mps' for Apple Silicon."
        },
    )
    style_tts_chunk_size: int = field(
        default=1024,
        metadata={
            "help": "Size of the audio data chunk processed per cycle, balancing playback latency and CPU load. Default is 1024."
        },
    )
    style_tts_sample_rate: int = field(
        default=24000,
        metadata={
            "help": "Sample rate for StyleTTS output audio in Hz. Default is 24000."
        },
    )
    style_tts_diffusion_steps: int = field(
        default=3,
        metadata={
            "help": "Number of diffusion steps for StyleTTS inference, affecting audio quality and performance. Default is 3."
        },
    )
