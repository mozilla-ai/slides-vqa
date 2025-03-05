import json


from slides_vqa.preprocess_video import split_video_into_chunks, merge_timestamps


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
