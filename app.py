import time
import cv2
import argparse
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict, deque
import torch


device = 'cuda' if torch.cuda.is_available() else "cpu"

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
MODEL_RESOLUTION = 640

SOURCE = np.array([
    [240, 169],
    [430, 177],
    [945, 480],
    [-94, 480]
])

TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1],
])

class ViewTransformer:

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

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

    model = YOLO(args.model).to(device)

    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    # Get Video Info
    video_info = sv.VideoInfo.from_video_path(video_path=args.source_video_path)

    #Creating Frames Generator using `sv`
    frame_generator = sv.get_video_frames_generator(source_path=args.source_video_path)
    # tracer initiation
    byte_track = sv.ByteTrack(
    frame_rate=video_info.fps, track_thresh=CONF_THRESHOLD
    )

    smoother = sv.DetectionsSmoother()
    
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
    color_annotator = sv.ColorAnnotator(
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER
    )
    polygon_zone = sv.PolygonZone(
        polygon=SOURCE,
        frame_resolution_wh=video_info.resolution_wh
    )

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    start = time.time()
    # Iterate over the Frames one by one
    for frame in frame_generator:
        frame = cv2.resize(frame, (720, 480))
        result = model(frame, imgsz=MODEL_RESOLUTION, verbose=True)[0]
        detections = sv.Detections.from_ultralytics(result)

        # filter out detections by class and confidence
        detections = detections[detections.confidence > CONF_THRESHOLD]
        detections = detections[detections.class_id != 0]

        # filter out detections outside the zone
        detections = detections[polygon_zone.trigger(detections)]

        # refine detections using non-max suppression
        detections = detections.with_nms(IOU_THRESHOLD)

        # pass detection through the tracker
        detections = byte_track.update_with_detections(detections=detections)
        detections = smoother.update_with_detections(detections=detections)
        points = detections.get_anchors_coordinates(
            anchor=sv.Position.BOTTOM_CENTER
        )

        # calculate the detections position inside the target RoI
        points = view_transformer.transform_points(points=points).astype(int)

        # store detections position
        for tracker_id, [_, y] in zip(detections.tracker_id, points):
            coordinates[tracker_id].append(y)

        # format labels
        labels = []
        for tracker_id in detections.tracker_id:
            if len(coordinates[tracker_id]) < video_info.fps / 2:
                labels.append(f"#{tracker_id}")
            else:
                # calculate speed
                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                distance = abs(coordinate_start - coordinate_end)
                time_taken = len(coordinates[tracker_id]) / video_info.fps
                speed = distance / time_taken * 3.6
                labels.append(f"#{tracker_id} {int(speed)} km/h")

        annotated_frame = frame.copy()
        annotated_frame = sv.draw_polygon(annotated_frame, polygon=SOURCE, color=sv.Color.RED, thickness=2)
        annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = color_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        end = time.time()
        fps = int(100 / (end - start)) if (end-start) > 0 else  0
        cv2.putText(annotated_frame, f'FPS: {fps}', (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Result', annotated_frame)
        if cv2.waitKey(1)  == ord('q'):  # press 'q' to quit
            break
    cv2.destroyAllWindows()

        