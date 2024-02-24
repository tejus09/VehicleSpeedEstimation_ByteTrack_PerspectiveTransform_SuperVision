import cv2
import numpy as np
import argparse
import supervision as sv
from inference.models.utils import get_roboflow_models

CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Argument Parser for Speed Estimation Python app")
    parser.add_argument("--source_video_path",
                        help="Path to the source video",
                        str=True,
                        required=True,
                        )
    parser.add_argument("--model", 
                        help="Specify the model to use",
                        default='yolov8n-640',
                        str=True,
                        required=True,
                        )
    return parser.parse_args()

if __name__ == "__main__":
    args = args_parser()
    
    SOURCE_VIDEO_PATH = args.source_video_path
    MODEL_NAME = args.model

    model = get_roboflow_models(MODEL_NAME)

    # Creating Frames Generator using `sv`
    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

    # Iterate over the Frames
    # for frame in frame_generator:

        