import tempfile
from pathlib import Path

import streamlit as st

from slides_vqa.preprocess_video import split_video_into_chunks

st.title("Slides Visual Question Answering")

input_video = st.file_uploader("Input Video", type="mp4")

if input_video:
    output_dir = Path(f"/tmp/chunks/{Path(input_video.name).stem}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with st.spinner("Splitting video into chunks..."):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_file:
            tmp_file.write(input_video.read())
            split_video_into_chunks(tmp_file.name, str(output_dir), 60)

    n_chunks = len(list(output_dir.glob("*.mp4")))
    st.success(f"Split video into {n_chunks} chunks")
