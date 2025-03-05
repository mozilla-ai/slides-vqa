SLIDE_TIMESTAMPS_PROMPT = """
Given the input video, return a list of timestamps with the different slides used in the presentation.
Use the following format:

[
    {"timestamp": "01:00", "title": "First slide", "description": "This slide displays the outline of the talk."},
    ...
]

Make the descriptions long enough to identify what is the relevant slide when the user ask a question about the video.
Only include an entry in the list when there is a change in the slides.
"""
