import os
import time

import google.generativeai as genai
from loguru import logger


class SmolVLM2:
    def __init__(self, model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.dtype = torch.bfloat16
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
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
        ).to(self.model.device, dtype=self.dtype)
        generated_ids = self.model.generate(**inputs, do_sample=False)
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        return generated_texts[0]


class Gemini:
    def __init__(self, model_id: str = "gemini-2.0-flash", **generation_config):
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.model = genai.GenerativeModel(
            model_name=model_id,
            generation_config=generation_config,
        )

    @staticmethod
    def upload_to_gemini(path, mime_type=None):
        logger.info(f"Uploading file '{path}'...")
        file = genai.upload_file(path, mime_type=mime_type)
        logger.info(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file

    @staticmethod
    def wait_for_file_active(file):
        logger.info("Waiting for file processing...")
        file = genai.get_file(file.name)
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(2)
            file = genai.get_file(file.name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
        logger.info("...all files ready")

    def process_video(self, input_video: str, prompt: str):
        file = self.upload_to_gemini(input_video, mime_type="video/mp4")
        self.wait_for_file_active(file)
        chat_session = self.model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        file,
                    ],
                },
                {
                    "role": "user",
                    "parts": [prompt],
                },
            ]
        )
        response = chat_session.send_message("INSERT_INPUT_HERE")
        return response.text
