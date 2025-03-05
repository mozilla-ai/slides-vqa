import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


class SmolVLM2:
    def __init__(self, model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",
        )

    def process_video(self, input_video: str, prompt: str):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": input_video},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)
        generated_ids = self.model.generate(**inputs, do_sample=False)
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        return generated_texts[0]
