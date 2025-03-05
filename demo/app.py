from pathlib import Path

import gradio as gr

from slides_vqa.preprocess_video import split_video_into_chunks


def process_video(input_video):
    if input_video is None:
        raise gr.Error("Please upload a video file.")

    input_video_path = Path(input_video)
    output_dir = Path(f"/tmp/chunks/{input_video_path.stem}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        split_video_into_chunks(str(input_video_path), str(output_dir), 60)
    except Exception as e:
        raise gr.Error(f"Error processing video: {e}")

    n_chunks = len(list(output_dir.glob("*.mp4")))
    yield f"Split video into {n_chunks} chunks"


with gr.Blocks() as demo:
    gr.Markdown("# Slides Visual Question Answering")
    with gr.Row():
        input_video = gr.Video(label="Input Video")
    with gr.Row():
        process_button = gr.Button("Process Video")
    with gr.Row():
        output_message = gr.Textbox(label="Output Message")
    process_button.click(fn=process_video, inputs=input_video, outputs=output_message)

demo.launch()
