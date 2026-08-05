#!/usr/bin/env python3
"""
App 4/4 (Inigo Arriazu): Video Processing Pipeline (multi-stage workflow).

Four dependent stages representing a video pipeline:
  Stage0 - Segment: split the video into as many segments as workers (plan).
  Stage1 - Extract: each worker "extracts" the frames of its segment.
  Stage2 - Enhance: each worker applies a filter (convolution) to its frames.
  Stage3 - Analyze: each worker computes features (brightness, edges) per frame.

This is CPU + memory (image arrays). To keep it self-contained and avoid needing a
real video or OpenCV, frames are GENERATED synthetically with numpy (deterministic
per seed). For real video: replace _frames_for_segment() with cv2.VideoCapture
reads. Requires: numpy.

    python examples/energy_apps/app_video.py
"""
from energy_report import profile_pipeline

TOTAL_FRAMES = 240
FRAME_H = 240
FRAME_W = 320


def _frames_for_segment(seg):
    """Deterministically generate the segment's frames as RGB arrays."""
    import numpy as np
    start, end = seg['start'], seg['end']
    rng = np.random.default_rng(seg['seg_id'])
    return np.clip(rng.normal(128, 40, size=(end - start, FRAME_H, FRAME_W, 3)), 0, 255).astype('uint8')


def stage0_segment(workers):
    """Split the video into 'workers' frame segments (plan)."""
    per = TOTAL_FRAMES // workers
    segs = []
    for i in range(workers):
        start = i * per
        end = TOTAL_FRAMES if i == workers - 1 else start + per
        segs.append({'seg_id': i, 'start': start, 'end': end})
    return segs


def stage1_extract(seg):
    """Extract (generate) the segment's frames and report the count."""
    frames = _frames_for_segment(seg)
    return {'seg_id': seg['seg_id'], 'n_frames': int(frames.shape[0])}


def stage2_enhance(seg):
    """Apply a sharpen/blur (3x3 convolution) to each frame of the segment."""
    import numpy as np
    frames = _frames_for_segment(seg)
    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype='float32')  # sharpen
    out = 0.0
    for fr in frames:
        gray = fr.mean(axis=2)
        # manual 3x3 convolution (CPU)
        acc = np.zeros_like(gray)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                acc += k[dy + 1, dx + 1] * np.roll(np.roll(gray, dy, 0), dx, 1)
        out += float(np.abs(acc).mean())
    return {'seg_id': seg['seg_id'], 'sharpness_sum': out}


def stage3_analyze(seg):
    """Compute per-frame features: mean brightness and edge density."""
    import numpy as np
    frames = _frames_for_segment(seg)
    brightness = float(frames.mean())
    edges = 0.0
    for fr in frames:
        gray = fr.mean(axis=2)
        gx = np.abs(np.diff(gray, axis=1)).mean()
        gy = np.abs(np.diff(gray, axis=0)).mean()
        edges += float(gx + gy)
    return {'seg_id': seg['seg_id'], 'brightness': brightness, 'edge_density': edges / len(frames)}


def run_pipeline(fexec, workers):
    """Run the 4 stages and return ALL futures produced."""
    f0 = fexec.map(stage0_segment, [workers])
    segs = fexec.get_result(fs=f0)[0]

    f1 = fexec.map(stage1_extract, [{'seg': s} for s in segs])
    fexec.get_result(fs=f1)

    f2 = fexec.map(stage2_enhance, [{'seg': s} for s in segs])
    fexec.get_result(fs=f2)

    f3 = fexec.map(stage3_analyze, [{'seg': s} for s in segs])
    fexec.get_result(fs=f3)

    return list(f0) + list(f1) + list(f2) + list(f3)


if __name__ == '__main__':
    # Local: sweep workers only (memory isn't enforced on localhost).
    # For AWS/K8s, add more memory values to the tuple below.
    MEMORY = [1024]
    config_space = [
        {'workers': w, 'memory': m}
        for m in MEMORY
        for w in (1, 2, 4, 8)
    ]
    profile_pipeline('video_processing', run_pipeline, config_space)