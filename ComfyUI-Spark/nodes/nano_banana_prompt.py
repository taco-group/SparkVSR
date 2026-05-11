"""SparkVSR Nano-Banana Pro prompt node."""

DEFAULT_NANO_BANANA_PRO_PROMPT = (
    "Super-Resolution and Restoration Task: Upscale this low-resolution image to "
    "high definition. Restore sharpness and clarity by removing degradation, blur, "
    "noise, grain, and compression artifacts."
)


class SparkVSRNanoBananaPrompt:
    """Standalone text prompt node for Nano-Banana Pro API references."""

    CATEGORY = "SparkVSR"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("api_prompt",)
    FUNCTION = "build_prompt"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "default": DEFAULT_NANO_BANANA_PRO_PROMPT,
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": (
                            "Prompt sent to fal-ai/nano-banana-pro/edit when "
                            "SparkVSR Prepare Reference uses ref_mode=nano-banana-pro-ref."
                        ),
                    },
                ),
            },
        }

    def build_prompt(self, prompt: str = ""):
        return (prompt or "",)
