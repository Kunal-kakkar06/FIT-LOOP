"""
scripts/inspect_waist_abdomen.py
--------------------------------
Diagnoses waist and abdomen underestimation on P001 and P002.
Extracts geometry, landmark positions, scanline profiles across ±100px of waist/abdomen,
and tests ellipse perimeter dynamics.
"""

from __future__ import annotations
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
from fitloop_measurement.measurements.slice_selector import select_anatomical_slice_from_mask
from fitloop_measurement.measurements.schemas import AnatomicalLevel
from fitloop_measurement.measurements.circumference import ramanujan_ellipse_perimeter
from fitloop_measurement.features.silhouette_geometry import torso_aware_scanline_width


def main():
    sessions = load_ground_truth_csv("pilot_template.csv")
    pose_ext = PoseExtractor()
    seg = BodySegmenter()
    feat_ext = GeometricFeatureExtractor()

    for s in sessions:
        print(f"\n=======================================================")
        print(f"PARTICIPANT: {s.participant_id} ({s.session_id})")
        print(f"Stature: {s.standing_height_cm} cm")
        print(f"Front image: {s.front_image_path}")
        print(f"Side image:  {s.side_image_path}")
        print(f"=======================================================")

        f_img = cv2.imread(s.front_image_path)
        s_img = cv2.imread(s.side_image_path)

        fp = pose_ext.extract(f_img, "front")
        sp = pose_ext.extract(s_img, "side")
        fs = seg.segment(f_img, "front")
        ss = seg.segment(s_img, "side")

        ff = feat_ext.extract_view_features(fp, fs)
        sf = feat_ext.extract_view_features(sp, ss)

        f_scale = s.standing_height_cm / float(ff.silhouette_height)
        s_scale = s.standing_height_cm / float(sf.silhouette_height)

        print(f"Silhouettes:")
        print(f"  Front: bbox y={ff.silhouette_bbox_y}, h={ff.silhouette_height} px -> scale = {f_scale:.5f} cm/px")
        print(f"  Side:  bbox y={sf.silhouette_bbox_y}, h={sf.silhouette_height} px -> scale = {s_scale:.5f} cm/px")

        # Check landmarks
        l_sh = ff.get_landmark_by_name("LEFT_SHOULDER")
        r_sh = ff.get_landmark_by_name("RIGHT_SHOULDER")
        l_hip = ff.get_landmark_by_name("LEFT_HIP")
        r_hip = ff.get_landmark_by_name("RIGHT_HIP")

        sh_y = (l_sh.pixel_y + r_sh.pixel_y) / 2.0
        hip_y = (l_hip.pixel_y + r_hip.pixel_y) / 2.0
        torso_span = hip_y - sh_y
        print(f"\nFront Torso Landmarks:")
        print(f"  Shoulder y: {sh_y:.1f}, Hip y: {hip_y:.1f}, Torso span: {torso_span:.1f} px")
        print(f"  Inter-shoulder width: {abs(r_sh.pixel_x - l_sh.pixel_x):.1f} px")
        print(f"  Inter-hip width:      {abs(r_hip.pixel_x - l_hip.pixel_x):.1f} px")

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
        print(f"Side Torso Landmarks:")
        print(f"  Shoulder y: {sh_y_s}, Hip y: {hip_y_s}, Torso span: {torso_span_s}")

        # Check currently selected slices
        for lvl in [AnatomicalLevel.CHEST, AnatomicalLevel.WAIST, AnatomicalLevel.ABDOMEN, AnatomicalLevel.HIPS]:
            f_slice, f_res = select_anatomical_slice_from_mask(lvl, ff, fs.mask)
            s_slice, s_res = select_anatomical_slice_from_mask(lvl, sf, ss.mask)
            ref_val = s.measurements[lvl.value].reference_value_cm

            fw_px = f_slice.width_pixels
            sd_px = s_slice.width_pixels
            fw_cm = fw_px * f_scale
            sd_cm = sd_px * s_scale
            pred_circ = ramanujan_ellipse_perimeter(fw_cm / 2.0, sd_cm / 2.0)
            err = pred_circ - ref_val

            print(f"\nLEVEL {lvl.value.upper()}: Ref={ref_val:.1f} cm | Pred={pred_circ:.2f} cm | Error={err:+.2f} cm")
            print(f"  Front: target_y={f_slice.image_y} (rel_sil={(f_slice.image_y - ff.silhouette_bbox_y)/ff.silhouette_height:.3f}), width={fw_px} px ({fw_cm:.1f} cm), segs={f_slice.segment_count}")
            print(f"  Side:  target_y={s_slice.image_y} (rel_sil={(s_slice.image_y - sf.silhouette_bbox_y)/sf.silhouette_height:.3f}), depth={sd_px} px ({sd_cm:.1f} cm), segs={s_slice.segment_count}")

        # Sweep scanlines around waist and abdomen for P002
        print(f"\n--- Vertical Sweep around Waist & Abdomen (Front & Side) ---")
        f_sh_y = int(round(sh_y))
        f_hip_y = int(round(hip_y))
        
        # sample at fractions 0.50 to 0.95 of torso span
        print(f"Fraction | Front_y | Front_w_cm | Side_y | Side_d_cm | Ellipse_Circ | Delta_from_RefWaist({s.measurements['waist'].reference_value_cm})")
        for frac in np.arange(0.50, 0.95, 0.05):
            curr_fy = int(round(sh_y + frac * torso_span))
            curr_sy = int(round(sh_y_s + frac * torso_span_s)) if (sh_y_s and torso_span_s) else int(round(sf.silhouette_bbox_y + (curr_fy - ff.silhouette_bbox_y)/ff.silhouette_height * sf.silhouette_height))
            
            # evaluate front width at curr_fy
            twr_f = torso_aware_scanline_width(
                binary_mask=fs.mask, image_y=curr_fy,
                shoulder_mid_x=(l_sh.pixel_x + r_sh.pixel_x)/2.0,
                hip_mid_x=(l_hip.pixel_x + r_hip.pixel_x)/2.0,
                shoulder_width=abs(r_sh.pixel_x - l_sh.pixel_x),
                hip_width=abs(r_hip.pixel_x - l_hip.pixel_x),
            )
            # evaluate side width at curr_sy
            twr_s = torso_aware_scanline_width(binary_mask=ss.mask, image_y=curr_sy)
            
            w_cm = (twr_f.width_pixels or 0) * f_scale
            d_cm = (twr_s.width_pixels or 0) * s_scale
            circ = ramanujan_ellipse_perimeter(w_cm / 2.0, d_cm / 2.0) if (w_cm > 0 and d_cm > 0) else 0.0
            print(f"  {frac:.2f}   |   {curr_fy:4d}  |   {w_cm:5.1f}    |  {curr_sy:4d}  |   {d_cm:5.1f}   |    {circ:6.2f}    |  {circ - s.measurements['waist'].reference_value_cm:+.2f}")


if __name__ == "__main__":
    main()
