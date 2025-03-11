import pymupdf


def extract_slides(input_presentation, output_dir):
    pdf = pymupdf.open(input_presentation)
    slides = []
    for n, slide in enumerate(pdf):
        pixmap = slide.get_pixmap()
        output_slide = f"{output_dir}/slide_{n}.png"
        pixmap.save(output_slide)
        slides.append(output_slide)
    return slides