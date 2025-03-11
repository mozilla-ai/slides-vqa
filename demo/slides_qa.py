from pathlib import Path
import gradio as gr
import spaces

from slides_vqa.models import SmolVLM2
from slides_vqa.preprocess_presentation import extract_slides

model = SmolVLM2()


def process_presentation(input_presentation):
    if input_presentation is None:
        raise gr.Error("Please upload a presentation file.")
    output_dir = Path(f"/tmp/slides/{Path(input_presentation).stem}")
    slides = extract_slides(input_presentation, output_dir)
    return gr.update(value=slides, visible=True)


@spaces.GPU
def answer_questions_about_slides(message, history, slides):
    if not slides:
        raise gr.Error("Please click in Process Presentation first.")
    return model.process_images([slide[0] for slide in slides], message)


with gr.Blocks() as demo:
    gr.Markdown("# Slides Visual Question Answering")
    with gr.Row():
        input_video = gr.File(label="Input Presentation", file_types=[".pdf"])
    with gr.Row():
        process_button = gr.Button("Process Presentation")
    with gr.Row():
        slides = gr.Gallery(label="Slides", show_label=True, visible=False)

    chat = gr.ChatInterface(
        answer_questions_about_slides,
        type="messages",
        title="Ask anything about the Slides",
        additional_inputs=[slides],
    )

    process_button.click(
        fn=process_presentation,
        inputs=input_video,
        outputs=[slides],
    )


demo.launch()
