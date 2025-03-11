import gradio as gr
import spaces
from pdf2image import convert_from_path

from slides_vqa.models import SmolVLM2


model = SmolVLM2()


def process_presentation(input_presentation):
    if input_presentation is None:
        raise gr.Error("Please upload a presentation file.")
    slides = convert_from_path(input_presentation)
    resized_slides = [
        slide.resize((640, 360)) for slide in slides
    ]
    return gr.update(value=resized_slides, visible=True)


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
