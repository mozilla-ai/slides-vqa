import json
from pathlib import Path

from slides_vqa.preprocess_video import (
    extract_slides,
    merge_timestamps,
    split_video_into_chunks,
)


def test_split_video_into_chunks(example_data, tmp_path):
    split_video_into_chunks(
        input_video=str(example_data / "The Transformer architecture.mp4"),
        output_dir=tmp_path,
        chunk_seconds=60,
    )
    assert len(list(tmp_path.glob("*.mp4"))) == 3


def test_merge_timestamps(tmp_path):
    (tmp_path / "001.json").write_text(
        json.dumps(
            [
                {"timestamp": "00:01", "title": "First slide", "description": "FOO."},
                {"timestamp": "00:03", "title": "Second slide", "description": "FOO."},
            ]
        )
    )
    (tmp_path / "002.json").write_text(
        json.dumps(
            [
                {"timestamp": "00:01", "title": "Third slide", "description": "BAR."},
            ]
        )
    )
    (tmp_path / "003.json").write_text(
        json.dumps(
            [
                {"timestamp": "00:00", "title": "Fourth slide", "description": "BAZ."},
            ]
        )
    )

    merged_timestamps = merge_timestamps(tmp_path, 5)
    assert merged_timestamps == [
        {"timestamp": "00:01", "title": "First slide", "description": "FOO."},
        {"timestamp": "00:03", "title": "Second slide", "description": "FOO."},
        {"timestamp": "00:06", "title": "Third slide", "description": "BAR."},
        {"timestamp": "00:10", "title": "Fourth slide", "description": "BAZ."},
    ]


def test_extract_slides(example_data, tmp_path):
    slides = extract_slides(
        input_video=str(example_data / "The Transformer architecture.mp4"),
        merged_timestamps=[
            {
                "timestamp": "00:03",
                "title": "The Transformer architecture",
                "description": "FOO.",
            },
            {
                "timestamp": "00:33",
                "title": "Attention Is All You Need",
                "description": "BAR.",
            },
        ],
        output_dir=tmp_path,
    )
    assert slides == [
        str(tmp_path / "00:03 - The Transformer architecture.jpg"),
        str(tmp_path / "00:33 - Attention Is All You Need.jpg"),
    ]
    for slide in slides:
        assert Path(slide).exists()
