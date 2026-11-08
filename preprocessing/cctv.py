"""Convert CCTV videos into the existing fi/no JPEG dataset format."""
import argparse
import json
from pathlib import Path
from run_output import artifact_path, create_run_dir


def read_json(file_path):
    with open(file_path, encoding="utf8") as stream:
        return json.load(stream)


def extract_clip(video_path, output_dir, start, end, size=(320, 240)):
    """Decode from zero to keep frame indices aligned even with codec seeking."""
    import cv2

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Remove only converter-generated files so shorter reruns have no stale frames.
    for frame in output_dir.glob("image_*.jpg"):
        if frame.stem.removeprefix("image_").isdigit():
            frame.unlink()
    (output_dir / "n_frames").unlink(missing_ok=True)
    video = cv2.VideoCapture(str(video_path))
    count = 0
    try:
        if not video.isOpened():
            raise OSError(f"Cannot open video: {video_path}")
        for frame_index in range(end):
            success, image = video.read()
            if not success or image is None:
                break
            if frame_index < start:
                continue
            image = cv2.resize(image, size)
            target = output_dir / f"image_{count + 1:05}.jpg"
            if not cv2.imwrite(str(target), image):
                raise OSError(f"Cannot write frame: {target}")
            count += 1
    finally:
        video.release()
    (output_dir / "n_frames").write_text(str(count), encoding="utf8")
    return count


def convert_cctv_dataset(json_path, video_dir, output_dir, max_videos=None):
    """Preserve the original first-segment policy and 350-frame clip cap."""
    database = read_json(json_path)["database"]
    converted = {}
    for index, (video_name, metadata) in enumerate(database.items()):
        if max_videos is not None and index >= max_videos:
            break
        annotations = metadata["annotations"]
        if not annotations:
            continue
        fps = float(metadata["frame_rate"])
        if fps <= 0:
            raise ValueError(f"Invalid frame rate for {video_name}: {fps}")
        begin, end = annotations[0]["segment"]
        begin, end = int(begin * fps), int(end * fps)
        if begin < 0 or end <= begin:
            raise ValueError(f"Invalid segment for {video_name}: {begin}, {end}")
        end = min(end, begin + 350)
        source = "CCTV_DATA" if metadata["source"] == "CCTV" else "NON_CCTV_DATA"
        video_path = Path(video_dir) / source / metadata["subset"] / f"{video_name}.mpeg"
        intervals = {"no": (max(0, begin - (end - begin)), begin), "fi": (begin, end)}
        for label, (start, stop) in intervals.items():
            name = f"{video_name}_{label}"
            count = extract_clip(video_path, Path(output_dir) / label / name, start, stop)
            if count:
                converted[name] = {
                    "subset": metadata["subset"], "annotations": {"label": label}
                }
    return {"labels": ["fi", "no"], "database": converted}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("cctv_jpg"),
                        help="Frame directory relative to the new output run folder")
    parser.add_argument("--output-json", type=Path, default=Path("cctv_jpg.json"),
                        help="Annotation file relative to the new output run folder")
    parser.add_argument("--max-videos", type=int)
    args = parser.parse_args()
    if args.max_videos is not None and args.max_videos < 1:
        parser.error("--max-videos must be positive")
    # Validate destinations before creating any run artifacts.
    from run_output import OUTPUT_ROOT
    try:
        artifact_path(OUTPUT_ROOT, args.output_dir)
        artifact_path(OUTPUT_ROOT, args.output_json)
    except ValueError as exc:
        parser.error(str(exc))
    run_dir = create_run_dir("convert_cctv")
    print(f"Run output: {run_dir}")
    output_dir = artifact_path(run_dir, args.output_dir)
    output_json = artifact_path(run_dir, args.output_json)
    result = convert_cctv_dataset(
        args.annotations, args.video_dir, output_dir, args.max_videos
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf8")
    return run_dir
