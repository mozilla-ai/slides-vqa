from pathlib import Path

import gradio as gr

from slides_vqa.models import Gemini
from slides_vqa.preprocess_video import (
    extract_slides,
    find_slide_timestamps,
    merge_timestamps,
    split_video_into_chunks,
)


def process_video(input_video):
    if input_video is None:
        raise gr.Error("Please upload a video file.")

    input_video_path = Path(input_video)
    output_dir = Path(f"/tmp/chunks/{input_video_path.stem}")
    output_dir.mkdir(parents=True, exist_ok=True)

    yield "Splitting video into chunks", None, None
    try:
        split_video_into_chunks(str(input_video_path), str(output_dir), 180)
    except Exception as e:
        raise gr.Error(f"Error processing video: {e}")

    n_chunks = len(list(output_dir.glob("*.mp4")))
    yield f"Split video into {n_chunks} chunks", None, None

    yield "Extracting slide timestamps...", None, None
    model = Gemini(response_mime_type="application/json")
    for chunk in find_slide_timestamps(model, output_dir):
        yield chunk, None, None

    yield "Merging timestamps...", None, None
    merged_timestamps = merge_timestamps(output_dir, 300)
    yield "Done!", merged_timestamps, None

    yield "Extracting slides", merged_timestamps, None
    slides = extract_slides(
        str(input_video_path), merged_timestamps, str(output_dir), offset=2
    )
    yield "Done!", merged_timestamps, slides


with gr.Blocks() as demo:
    gr.Markdown("# Slides Visual Question Answering")
    with gr.Row():
        input_video = gr.Video(label="Input Video")
    with gr.Row():
        process_button = gr.Button("Process Video")
    with gr.Row():
        output_message = gr.Textbox(label="Status")
    with gr.Row():
        slide_timestamps = gr.JSON(label="Extracted Slide Timestamps")
        slides = gr.Gallery(label="Slides", show_label=False)
    process_button.click(
        fn=process_video,
        inputs=input_video,
        outputs=[output_message, slide_timestamps, slides],
    )

demo.launch()
