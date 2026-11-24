"""
scripts/generate_waist_abdomen_overlays.py
-----------------------------------------
Generates detailed multi-scanline diagnostic overlays for waist and abdomen
on both participants (P001 and P002), showing:
- Original image + segmentation contour
- Pose landmarks (shoulders, hips, elbows, wrists)
- Torso window bounds [x_min, x_max]
- Scanline levels for 50%, 55%, 60%, 65% (waist), 70%, 75%, 80% (abdomen), 85%, 90%
- Selected torso run vs rejected arm runs
"""

import sys
from pathlib import Path
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import cv2
import numpy as np
from fitloop_measurement.evaluation.dataset import load_ground_truth_csv
from fitloop_measurement.pose import PoseExtractor
from fitloop_measurement.segmentation import BodySegmenter
from fitloop_measurement.features import GeometricFeatureExtractor
from fitloop_measurement.features.silhouette_geometry import torso_aware_scanline_width


def main():
    sessions = load_ground_truth_csv("pilot_template.csv")
    pose_ext = PoseExtractor()
    seg = BodySegmenter()
    feat_ext = GeometricFeatureExtractor()

    out_dir = Path("diagnostic_overlays/stage7c")
    out_dir.mkdir(parents=True, exist_ok=True)

    for s in sessions:
        p_id = s.participant_id
        f_img = cv2.imread(s.front_image_path)
        s_img = cv2.imread(s.side_image_path)

        fp = pose_ext.extract(f_img, "front")
        sp = pose_ext.extract(s_img, "side")
        fs = seg.segment(f_img, "front")
        ss = seg.segment(s_img, "side")

        ff = feat_ext.extract_view_features(fp, fs)
        sf = feat_ext.extract_view_features(sp, ss)

        # Front landmarks
        l_sh = ff.get_landmark_by_name("LEFT_SHOULDER")
        r_sh = ff.get_landmark_by_name("RIGHT_SHOULDER")
        l_hip = ff.get_landmark_by_name("LEFT_HIP")
        r_hip = ff.get_landmark_by_name("RIGHT_HIP")

        sh_y = (l_sh.pixel_y + r_sh.pixel_y) / 2.0
        hip_y = (l_hip.pixel_y + r_hip.pixel_y) / 2.0
        torso_span = hip_y - sh_y
        sh_w = abs(r_sh.pixel_x - l_sh.pixel_x)
        hip_w = abs(r_hip.pixel_x - l_hip.pixel_x)

        # Draw front visualization
        vis_f = f_img.copy()
        # Draw torso window vertical guide lines
        # Window center:
        cx = (l_sh.pixel_x + r_sh.pixel_x + l_hip.pixel_x + r_hip.pixel_x) / 4.0
        half_w = max(sh_w, hip_w) / 2.0 * 1.25
        cv2.line(vis_f, (int(cx - half_w), int(sh_y)), (int(cx - half_w), int(hip_y)), (255, 200, 0), 2)
        cv2.line(vis_f, (int(cx + half_w), int(sh_y)), (int(cx + half_w), int(hip_y)), (255, 200, 0), 2)

        fractions = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
        colors = {
            0.65: (0, 255, 0),    # Current Waist: Bright Green
            0.80: (0, 255, 255),  # Current Abdomen: Bright Yellow
        }

        for frac in fractions:
            y = int(round(sh_y + frac * torso_span))
            twr = torso_aware_scanline_width(
                binary_mask=fs.mask, image_y=y,
                shoulder_mid_x=(l_sh.pixel_x + r_sh.pixel_x)/2.0,
                hip_mid_x=(l_hip.pixel_x + r_hip.pixel_x)/2.0,
                shoulder_width=sh_w, hip_width=hip_w,
            )
            col = colors.get(frac, (180, 180, 180))
            thickness = 3 if frac in [0.65, 0.80] else 1

            # Draw all intervals
            for start_x, end_x in twr.all_intervals:
                if twr.selected_interval is not None and (start_x, end_x) == twr.all_intervals[twr.selected_interval]:
                    cv2.line(vis_f, (start_x, y), (end_x, y), col, thickness + 1)
                else:
                    cv2.line(vis_f, (start_x, y), (end_x, y), (0, 0, 255), 1) # Red for rejected

            tag = "WAIST (65%)" if frac == 0.65 else ("ABDOMEN (80%)" if frac == 0.80 else f"{int(frac*100)}%")
            cv2.putText(vis_f, f"{tag}: w={twr.width_pixels}px", (10, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        # Draw Side visualization
        vis_s = s_img.copy()
        # Side landmarks
        l_sh_s = sf.get_landmark_by_name("LEFT_SHOULDER")
        r_sh_s = sf.get_landmark_by_name("RIGHT_SHOULDER")
        l_hip_s = sf.get_landmark_by_name("LEFT_HIP")
        r_hip_s = sf.get_landmark_by_name("RIGHT_HIP")
        sh_ys = [lm.pixel_y for lm in (l_sh_s, r_sh_s) if lm and lm.is_valid and lm.pixel_y is not None]
        hip_ys = [lm.pixel_y for lm in (l_hip_s, r_hip_s) if lm and lm.is_valid and lm.pixel_y is not None]
        sh_y_s = sum(sh_ys) / len(sh_ys) if sh_ys else None
        hip_y_s = sum(hip_ys) / len(hip_ys) if hip_ys else None
        torso_span_s = (hip_y_s - sh_y_s) if (sh_y_s and hip_y_s) else None

        for frac in fractions:
            y = int(round(sh_y_s + frac * torso_span_s)) if (sh_y_s and torso_span_s) else int(round(sf.silhouette_bbox_y + frac * sf.silhouette_height))
            twr_s = torso_aware_scanline_width(binary_mask=ss.mask, image_y=y)
            col = colors.get(frac, (180, 180, 180))
            thickness = 3 if frac in [0.65, 0.80] else 1
            if twr_s.is_valid and twr_s.left_x is not None and twr_s.right_x is not None:
                cv2.line(vis_s, (twr_s.left_x, y), (twr_s.right_x, y), col, thickness + 1)
            tag = "WAIST (65%)" if frac == 0.65 else ("ABDOMEN (80%)" if frac == 0.80 else f"{int(frac*100)}%")
            cv2.putText(vis_s, f"{tag}: d={twr_s.width_pixels}px", (10, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        f_out_path = out_dir / f"{p_id}_front_waist_abdomen_sweep.jpg"
        s_out_path = out_dir / f"{p_id}_side_waist_abdomen_sweep.jpg"
        cv2.imwrite(str(f_out_path), vis_f)
        cv2.imwrite(str(s_out_path), vis_s)
        print(f"Saved: {f_out_path} and {s_out_path}")


if __name__ == "__main__":
    main()
