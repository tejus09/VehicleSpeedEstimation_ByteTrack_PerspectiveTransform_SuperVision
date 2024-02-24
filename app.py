import time
import cv2
import argparse
import numpy as np
import supervision as sv
from ultralytics import YOLO

CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
MODEL_RESOLUTION = 640

def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Argument Parser for Speed Estimation Python app")
    parser.add_argument("--source_video_path",
                        help="Path to the source video",
                        type=str,
                        required=True,
                        )
    parser.add_argument("--model", 
                        help="Specify the model to use",
                        default="yolov8n.pt",
                        type=str,
                        )
    return parser.parse_args()

if __name__ == "__main__":
    args = args_parser()

    model = YOLO(args.model)

    # Get Video Info
    video_info = sv.VideoInfo.from_video_path(video_path=args.source_video_path)

    #Creating Frames Generator using `sv`
    frame_generator = sv.get_video_frames_generator(source_path=args.source_video_path)

    # annotators configuration
    thickness = sv.calculate_dynamic_line_thickness(
        resolution_wh=(720, 480)
    )
    text_scale = sv.calculate_dynamic_text_scale(
        resolution_wh=(720, 480)
    )
    bounding_box_annotator = sv.BoundingBoxAnnotator(
        thickness=thickness
    )
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER
    )


    start = time.time()
    # Iterate over the Frames one by one
    for frame in frame_generator:
        frame = cv2.resize(frame, (720, 480))
        result = model(frame, imgsz=MODEL_RESOLUTION, verbose=False, device=0)[0]
        detections = sv.Detections.from_ultralytics(result)

        # filter out detections by class and confidence
        detections = detections[detections.confidence > CONF_THRESHOLD]
        detections = detections[detections.class_id != 0]

        annotated_frame = frame.copy()
        # annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = bounding_box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

        end = time.time()
        fps = int(100 / (end - start)) if (end-start) > 0 else  0
        cv2.putText(annotated_frame, f'FPS: {fps}', (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Result', annotated_frame)
        if cv2.waitKey(1)  == ord('q'):  # press 'q' to quit
            break
    cv2.destroyAllWindows()

        