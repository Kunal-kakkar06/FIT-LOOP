"""
scripts/stage7d_audit.py
------------------------
Stage 7D — Anatomical Slice Validation and Calibration Audit.

Tasks:
  1. Ground-truth independence audit
  2. Calibration audit (per-participant, per-view)
  3. Anatomical slice correspondence audit + overlays
  4. Error hypothesis separation
  5. Vertical sweep sensitivity validation
"""
from __future__ import annotations
import json, math, os, sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fitloop_measurement.evaluation.dataset import load_ground_truth_csv
from fitloop_measurement.features import GeometricFeatureExtractor
from fitloop_measurement.features.silhouette_geometry import torso_aware_scanline_width
from fitloop_measurement.measurements.calibration import calibrate_dual_views
from fitloop_measurement.measurements.circumference import ramanujan_ellipse_perimeter
from fitloop_measurement.measurements.schemas import AnatomicalLevel
from fitloop_measurement.measurements.slice_selector import select_anatomical_slice_from_mask
from fitloop_measurement.pose import PoseExtractor
from fitloop_measurement.segmentation import BodySegmenter

OUT_DIR = REPO_ROOT / "diagnostic_overlays" / "stage7d"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = REPO_ROOT / "pilot_template.csv"

LEVEL_FRACTIONS = {"chest": 0.35, "waist": 0.65, "abdomen": 0.80, "hips": 1.05}
LEVEL_ENUM_MAP = {
    "chest":   AnatomicalLevel.CHEST,
    "waist":   AnatomicalLevel.WAIST,
    "abdomen": AnatomicalLevel.ABDOMEN,
    "hips":    AnatomicalLevel.HIPS,
}

# ── Task 1 ───────────────────────────────────────────────────────────────────
def audit_ground_truth(sessions):
    report = {}
    print("\n" + "="*72)
    print("TASK 1 — Ground-Truth Independence Audit")
    print("="*72)
    for s in sessions:
        pid = s.participant_id
        flags = []
        for dim in ["chest","waist","abdomen","hips"]:
            m = s.measurements[dim]
            readings = m.readings_cm
            ref = m.reference_value_cm
            if len(set(readings)) == 1:
                flags.append(f"  [{dim.upper()}] All {len(readings)} readings IDENTICAL ({readings[0]} cm) — cannot verify independence")
            else:
                spread = max(readings)-min(readings)
                flags.append(f"  [{dim.upper()}] readings={readings} ref={ref:.1f}cm spread={spread:.1f}cm")
        print(f"\n{pid} ({s.session_id}) h={s.standing_height_cm}cm")
        for f in flags:
            print(f)
        report[pid] = flags
    print("\n[STATUS] P001 and P002 all three readings are identical => only a single tape")
    print("         entry is recorded per measurement site. Independence CANNOT BE VERIFIED.")
    print("         Evaluation accuracy claims are PRELIMINARY until repeated independent")
    print("         tape readings are documented separately.")
    return report

# ── Helpers ───────────────────────────────────────────────────────────────────
def _mid_y(a, b):
    if a and a.is_valid and b and b.is_valid: return (a.pixel_y + b.pixel_y)/2.0
    if a and a.is_valid: return a.pixel_y
    if b and b.is_valid: return b.pixel_y
    return None

def _mid_x(a, b):
    if a and a.is_valid and b and b.is_valid: return (a.pixel_x + b.pixel_x)/2.0
    if a and a.is_valid: return a.pixel_x
    if b and b.is_valid: return b.pixel_x
    return None

def _get_landmarks(feats):
    names = ["LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_HIP","RIGHT_HIP",
             "LEFT_ELBOW","RIGHT_ELBOW","LEFT_WRIST","RIGHT_WRIST"]
    return {n: feats.get_landmark_by_name(n) for n in names}

# ── Task 2 ───────────────────────────────────────────────────────────────────
def audit_calibration(sessions, pose_ex, seg_ex, feat_ex):
    print("\n"+"="*72)
    print("TASK 2 — Calibration Audit")
    print("="*72)
    calib_table = []
    for s in sessions:
        pid = s.participant_id
        h_cm = s.standing_height_cm
        for view_name, img_path in [("front",s.front_image_path),("side",s.side_image_path)]:
            img = cv2.imread(img_path)
            if img is None:
                print(f"  [{pid}][{view_name}] CANNOT LOAD: {img_path}")
                continue
            ih, iw = img.shape[:2]
            pose  = pose_ex.extract(img, view_type=view_name)
            seg   = seg_ex.segment(img, view_type=view_name)
            feats = feat_ex.extract_view_features(pose, seg)
            sil_h  = feats.silhouette_height or 0
            bbox_y = feats.silhouette_bbox_y or 0
            bot_y  = bbox_y + sil_h
            scale  = (h_cm / sil_h) if sil_h > 0 else None
            clip_t = bbox_y <= 2
            clip_b = bot_y >= (ih - 2)
            print(f"\n  [{pid}][{view_name}]  image={iw}x{ih}  sil: top={bbox_y} h={sil_h} bot={bot_y}")
            print(f"    scale={scale:.5f} cm/px" if scale else "    scale=unavailable")
            print(f"    clipped_top={clip_t}  clipped_bottom={clip_b}")
            print(f"    pose_ok={pose.is_successful}  seg_ok={seg.is_successful}")
            if feats.warnings:
                print(f"    feature_warnings={feats.warnings}")
            row = {
                "participant_id": pid, "view": view_name,
                "img_w": iw, "img_h": ih,
                "sil_bbox_y": bbox_y, "sil_height_px": sil_h, "sil_bottom_y": bot_y,
                "standing_height_cm": h_cm,
                "scale_cm_per_px": round(scale,5) if scale else None,
                "clipped_top": clip_t, "clipped_bottom": clip_b,
                "pose_ok": pose.is_successful, "seg_ok": seg.is_successful,
            }
            calib_table.append(row)

    print("\n  --- Cross-view scale comparison ---")
    by_pid: Dict = {}
    for r in calib_table:
        by_pid.setdefault(r["participant_id"],{})[r["view"]] = r.get("scale_cm_per_px")
    for pid, sv in by_pid.items():
        fs, ss = sv.get("front"), sv.get("side")
        if fs and ss:
            ratio = fs/ss
            diff = abs(ratio-1)*100
            flag = " ⚠ >15% divergence" if diff>15 else (" ⚠ >5%" if diff>5 else "")
            print(f"  {pid}: front={fs:.5f}  side={ss:.5f}  ratio={ratio:.4f} ({diff:.1f}%){flag}")
    return calib_table

# ── Task 3 ───────────────────────────────────────────────────────────────────
def _draw_overlay(img, mask, target_y, sel_int, all_ints, tw_range,
                  level_name, view_name, shoulder_y, hip_y, lm_overlay):
    out = img.copy()
    h, w = out.shape[:2]
    mc = np.zeros_like(out)
    mc[mask > 0] = (0,80,0)
    out = cv2.addWeighted(out, 0.75, mc, 0.25, 0)
    for (sx,ex) in all_ints:
        if sel_int and (sx,ex)==tuple(sel_int): continue
        cv2.line(out,(sx,target_y),(ex,target_y),(0,0,200),3)
    if sel_int:
        cv2.line(out,(sel_int[0],target_y),(sel_int[1],target_y),(0,255,80),4)
    if tw_range:
        for tx in tw_range:
            cv2.line(out,(tx,max(0,target_y-20)),(tx,min(h-1,target_y+20)),(255,200,0),2)
    cv2.line(out,(0,target_y),(w-1,target_y),(255,255,255),1)
    for (lx,ly,ln) in lm_overlay:
        cv2.circle(out,(int(lx),int(ly)),7,(0,255,255),-1)
        cv2.putText(out,ln,(int(lx)+8,int(ly)),cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,255,255),1,cv2.LINE_AA)
    for ry, col, lab in [(shoulder_y,(255,128,0),"SHL"),(hip_y,(0,128,255),"HIP")]:
        if ry is not None:
            cv2.line(out,(0,int(ry)),(w-1,int(ry)),col,1)
            cv2.putText(out,lab,(5,int(ry)-4),cv2.FONT_HERSHEY_SIMPLEX,0.42,col,1,cv2.LINE_AA)
    sw = (sel_int[1]-sel_int[0]+1) if sel_int else 0
    txt = f"{level_name.upper()} ({view_name}) y={target_y}  w={sw}px"
    cv2.putText(out,txt,(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.62,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(out,txt,(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.62,(30,30,30),1,cv2.LINE_AA)
    return out

def audit_anatomical_slices(sessions, pose_ex, seg_ex, feat_ex):
    print("\n"+"="*72)
    print("TASK 3 — Anatomical Slice Correspondence Audit")
    print("="*72)
    slice_table = []
    for s in sessions:
        pid = s.participant_id
        h_cm = s.standing_height_cm
        for view_name, img_path in [("front",s.front_image_path),("side",s.side_image_path)]:
            img = cv2.imread(img_path)
            if img is None: continue
            pose  = pose_ex.extract(img, view_type=view_name)
            seg   = seg_ex.segment(img, view_type=view_name)
            feats = feat_ex.extract_view_features(pose, seg)
            if not seg.is_successful:
                print(f"  [{pid}][{view_name}] seg failed — skip"); continue
            mask = seg.mask
            lms = _get_landmarks(feats)
            l_sh,r_sh,l_hip,r_hip = lms["LEFT_SHOULDER"],lms["RIGHT_SHOULDER"],lms["LEFT_HIP"],lms["RIGHT_HIP"]
            shoulder_y = _mid_y(l_sh, r_sh)
            hip_y      = _mid_y(l_hip, r_hip)
            torso_span = (hip_y-shoulder_y) if (shoulder_y and hip_y and hip_y>shoulder_y) else None
            sil_h  = feats.silhouette_height or 0
            bbox_y = feats.silhouette_bbox_y or 0
            scale  = (h_cm/sil_h) if sil_h>0 else None
            lm_overlay = []
            for ln in ["LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_HIP","RIGHT_HIP","LEFT_ELBOW","RIGHT_ELBOW"]:
                lm = feats.get_landmark_by_name(ln)
                if lm and lm.is_valid and lm.pixel_x is not None:
                    lm_overlay.append((lm.pixel_x, lm.pixel_y, ln[:4]))
            print(f"\n  [{pid}][{view_name}]  sh_y={shoulder_y:.1f}  hip_y={hip_y:.1f}  span={torso_span:.1f}px  scale={scale:.5f}cm/px" if (shoulder_y and hip_y and torso_span and scale) else f"\n  [{pid}][{view_name}]  landmarks or scale partial")
            for level_name, level_enum in LEVEL_ENUM_MAP.items():
                sc, sr = select_anatomical_slice_from_mask(level_enum, feats, mask)
                actual_y  = sc.image_y if sc else None
                actual_w  = sc.width_pixels if sc else None
                w_cm      = (actual_w * scale) if (actual_w and scale) else None
                rel_y     = sc.relative_y if sc else None
                ref_cm    = s.measurements[level_name].reference_value_cm
                exp_y = int(round(shoulder_y + LEVEL_FRACTIONS[level_name]*torso_span)) if (shoulder_y and torso_span) else None
                # re-run for overlay detail
                if actual_y is not None:
                    sh_mid_x  = _mid_x(l_sh,r_sh)  if view_name=="front" else None
                    hip_mid_x = _mid_x(l_hip,r_hip) if view_name=="front" else None
                    sh_w = abs(l_sh.pixel_x-r_sh.pixel_x) if (l_sh and l_sh.is_valid and r_sh and r_sh.is_valid and l_sh.pixel_x is not None and view_name=="front") else None
                    hip_w= abs(l_hip.pixel_x-r_hip.pixel_x) if (l_hip and l_hip.is_valid and r_hip and r_hip.is_valid and l_hip.pixel_x is not None and view_name=="front") else None
                    l_el,r_el = lms["LEFT_ELBOW"],lms["RIGHT_ELBOW"]
                    l_wr,r_wr = lms["LEFT_WRIST"],lms["RIGHT_WRIST"]
                    el_xs = [lm.pixel_x for lm in (l_el,r_el) if lm and lm.is_valid and lm.pixel_x is not None and view_name=="front"]
                    wr_xs = [lm.pixel_x for lm in (l_wr,r_wr) if lm and lm.is_valid and lm.pixel_x is not None and view_name=="front"]
                    twr = torso_aware_scanline_width(
                        binary_mask=mask, image_y=actual_y,
                        shoulder_mid_x=sh_mid_x, hip_mid_x=hip_mid_x,
                        elbow_xs=el_xs or None, wrist_xs=wr_xs or None,
                        silhouette_top_y=bbox_y, silhouette_height=sil_h,
                        shoulder_width=sh_w, hip_width=hip_w)
                    all_ints = twr.all_intervals
                    sel_int  = (twr.left_x, twr.right_x) if twr.is_valid else None
                    tw = twr.torso_window
                    tw_range = (tw.x_min, tw.x_max) if tw.is_valid else None
                else:
                    all_ints, sel_int, tw_range = [], None, None
                print(f"    [{level_name.upper():8s}] y={actual_y}(exp={exp_y}) rel={rel_y:.3f}  "
                      f"w={actual_w}px={w_cm:.1f}cm  ref={ref_cm:.1f}cm  method={sr.selection_method[:45]}"
                      if (actual_y is not None and rel_y is not None and actual_w is not None and w_cm is not None)
                      else f"    [{level_name.upper():8s}] UNAVAILABLE — {sr.warnings}")
                slice_table.append({
                    "participant_id":pid,"view":view_name,"level":level_name,
                    "shoulder_y":shoulder_y,"hip_y":hip_y,"torso_span":torso_span,
                    "expected_y":exp_y,"actual_y":actual_y,"rel_y":rel_y,
                    "width_px":actual_w,"width_cm":w_cm,"tape_ref_cm":ref_cm,
                    "selection_method":sr.selection_method,"warnings":sr.warnings,
                })
                if actual_y is not None:
                    ov = _draw_overlay(img,mask,actual_y,sel_int,all_ints,tw_range,
                                       level_name,view_name,shoulder_y,hip_y,lm_overlay)
                    fname = OUT_DIR / f"{pid}_{view_name}_{level_name}_slice.png"
                    cv2.imwrite(str(fname), ov)
    return slice_table

# ── Task 4 ───────────────────────────────────────────────────────────────────
def audit_error_hypotheses(sessions, slice_table, calib_table):
    print("\n"+"="*72)
    print("TASK 4 — Error Hypothesis Separation")
    print("="*72)
    cb = {(r["participant_id"],r["view"]):r for r in calib_table}
    for s in sessions:
        pid = s.participant_id
        print(f"\n  Participant: {pid}")
        fc = cb.get((pid,"front"),{})
        sc2= cb.get((pid,"side"),{})

        print("\n  A. Incorrect anatomical slice placement")
        for level in ["waist","abdomen"]:
            fr = next((r for r in slice_table if r["participant_id"]==pid and r["view"]=="front" and r["level"]==level),None)
            sr = next((r for r in slice_table if r["participant_id"]==pid and r["view"]=="side"  and r["level"]==level),None)
            if fr and sr:
                frel = fr.get("rel_y"); srel = sr.get("rel_y")
                if frel is not None and srel is not None:
                    print(f"    [{level}] front rel_y={frel:.3f}  side rel_y={srel:.3f}  Δ={abs(frel-srel):.3f}")
                    if abs(frel-srel) > 0.02:
                        print(f"           ⚠ Front/side scanlines sample different anatomical heights (Δ>{0.02:.0%} torso span)")

        print("\n  B. Calibration / perspective differences")
        fs, ss = fc.get("scale_cm_per_px"), sc2.get("scale_cm_per_px")
        fsh, ssh = fc.get("sil_height_px",0), sc2.get("sil_height_px",0)
        if fs and ss:
            ratio = fs/ss; diff = abs(ratio-1)*100
            print(f"    front_scale={fs:.5f}  side_scale={ss:.5f}  ratio={ratio:.4f} ({diff:.1f}%)")
            print(f"    front_sil_h={fsh}px  side_sil_h={ssh}px  Δsil_h={abs(fsh-ssh)}px")
            if diff>15:
                print("    ⚠ >15% scale discrepancy — significant framing/perspective difference.")
            elif diff>5:
                print("    ⚠ >5% scale discrepancy — minor framing difference detected.")
            else:
                print("    ✓ Scale discrepancy <5% — calibration consistent between views.")
        else:
            print("    Calibration data unavailable.")
        print("    NOTE: Independent scale per view does not correct for body lean,")
        print("          subject distance differences, or lens distortion.")

        print("\n  C. Segmentation boundary errors")
        print("    Status: PLAUSIBLE but unconfirmed from metadata alone.")
        print("    Action: Inspect overlay images — check mask boundary at waist/abdomen scanlines.")
        print("    Fitted vs. loose clothing and dark backgrounds may affect silhouette boundaries.")

        print("\n  D. Ellipse approximation deficit")
        for level in ["waist","abdomen","hips"]:
            ref_cm = s.measurements[level].reference_value_cm
            fr = next((r for r in slice_table if r["participant_id"]==pid and r["view"]=="front" and r["level"]==level),None)
            sr = next((r for r in slice_table if r["participant_id"]==pid and r["view"]=="side"  and r["level"]==level),None)
            if fr and sr and fr.get("width_cm") and sr.get("width_cm"):
                fw = fr["width_cm"]; sw = sr["width_cm"]
                a,b = fw/2.0, sw/2.0
                pred = ramanujan_ellipse_perimeter(a,b)
                # Stadium approximation (rectangle + two semicircles)
                dim_a,dim_b = max(a,b), min(a,b)
                stadium = math.pi*dim_b*2 + 2*2*(dim_a-dim_b)
                error = pred - ref_cm
                print(f"    [{level}] ref={ref_cm:.1f}  pred={pred:.2f}  error={error:+.2f}cm")
                print(f"           fw={fw:.1f}cm  sd={sw:.1f}cm  a={a:.2f}  b={b:.2f}")
                print(f"           ellipse_vs_stadium: {pred-stadium:+.2f}cm (stadium={'%.2f'%stadium}cm)")

# ── Task 5 ───────────────────────────────────────────────────────────────────
def audit_vertical_sweep(sessions, pose_ex, seg_ex, feat_ex):
    print("\n"+"="*72)
    print("TASK 5 — Vertical Sweep Sensitivity Validation")
    print("="*72)
    sweep_results = []
    for s in sessions:
        pid = s.participant_id
        h_cm = s.standing_height_cm
        f_img = cv2.imread(s.front_image_path)
        s_img = cv2.imread(s.side_image_path)
        if f_img is None or s_img is None: continue
        f_pose = pose_ex.extract(f_img, view_type="front")
        s_pose = pose_ex.extract(s_img, view_type="side")
        f_seg  = seg_ex.segment(f_img, view_type="front")
        s_seg  = seg_ex.segment(s_img, view_type="side")
        f_feats= feat_ex.extract_view_features(f_pose, f_seg)
        s_feats= feat_ex.extract_view_features(s_pose, s_seg)
        if not (f_seg.is_successful and s_seg.is_successful):
            print(f"  [{pid}] Seg failed"); continue
        f_mask,s_mask = f_seg.mask, s_seg.mask
        lf = _get_landmarks(f_feats); ls = _get_landmarks(s_feats)
        f_sh_y  = _mid_y(lf["LEFT_SHOULDER"],lf["RIGHT_SHOULDER"])
        f_hip_y = _mid_y(lf["LEFT_HIP"],lf["RIGHT_HIP"])
        s_sh_y  = _mid_y(ls["LEFT_SHOULDER"],ls["RIGHT_SHOULDER"])
        s_hip_y = _mid_y(ls["LEFT_HIP"],ls["RIGHT_HIP"])
        if None in (f_sh_y,f_hip_y,s_sh_y,s_hip_y):
            print(f"  [{pid}] Incomplete landmarks"); continue
        f_span = f_hip_y - f_sh_y
        s_span = s_hip_y - s_sh_y
        f_sil_h = f_feats.silhouette_height or 0
        s_sil_h = s_feats.silhouette_height or 0
        f_scale = (h_cm/f_sil_h) if f_sil_h>0 else None
        s_scale = (h_cm/s_sil_h) if s_sil_h>0 else None
        if not (f_scale and s_scale): continue
        f_sh_mid_x  = _mid_x(lf["LEFT_SHOULDER"],lf["RIGHT_SHOULDER"])
        f_hip_mid_x = _mid_x(lf["LEFT_HIP"],lf["RIGHT_HIP"])
        l_sh,r_sh = lf["LEFT_SHOULDER"],lf["RIGHT_SHOULDER"]
        l_hip,r_hip= lf["LEFT_HIP"],lf["RIGHT_HIP"]
        f_sh_w  = abs(l_sh.pixel_x-r_sh.pixel_x) if (l_sh and l_sh.is_valid and r_sh and r_sh.is_valid and l_sh.pixel_x is not None) else None
        f_hip_w = abs(l_hip.pixel_x-r_hip.pixel_x) if (l_hip and l_hip.is_valid and r_hip and r_hip.is_valid and l_hip.pixel_x is not None) else None
        el_xs = [lm.pixel_x for lm in (lf["LEFT_ELBOW"],lf["RIGHT_ELBOW"]) if lm and lm.is_valid and lm.pixel_x is not None]
        wr_xs = [lm.pixel_x for lm in (lf["LEFT_WRIST"],lf["RIGHT_WRIST"]) if lm and lm.is_valid and lm.pixel_x is not None]
        f_bbox_y = f_feats.silhouette_bbox_y or 0
        s_bbox_y = s_feats.silhouette_bbox_y or 0

        for level_name in ["waist","abdomen"]:
            base_frac = LEVEL_FRACTIONS[level_name]
            ref_cm = s.measurements[level_name].reference_value_cm
            sweep_fracs = [base_frac + d/100.0 for d in range(-12,13)]
            print(f"\n  [{pid}] SWEEP [{level_name.upper()}]  base_frac={base_frac}  ref={ref_cm:.1f}cm")
            print(f"    {'frac':>6}  {'f_y':>5}  {'s_y':>5}  {'fw_cm':>6}  {'sd_cm':>6}  {'circ_cm':>8}  {'error':>7}")
            best_err = float("inf"); best_row = None
            for frac in sweep_fracs:
                f_y = int(round(f_sh_y + frac*f_span))
                s_y = int(round(s_sh_y + frac*s_span))
                ft = torso_aware_scanline_width(
                    binary_mask=f_mask, image_y=f_y,
                    shoulder_mid_x=f_sh_mid_x, hip_mid_x=f_hip_mid_x,
                    elbow_xs=el_xs or None, wrist_xs=wr_xs or None,
                    silhouette_top_y=f_bbox_y, silhouette_height=f_sil_h,
                    shoulder_width=f_sh_w, hip_width=f_hip_w)
                st = torso_aware_scanline_width(
                    binary_mask=s_mask, image_y=s_y,
                    silhouette_top_y=s_bbox_y, silhouette_height=s_sil_h)
                if not (ft.is_valid and st.is_valid): continue
                fw_cm = ft.width_pixels * f_scale
                sd_cm = st.width_pixels * s_scale
                try: circ = ramanujan_ellipse_perimeter(fw_cm/2, sd_cm/2)
                except: continue
                error = circ - ref_cm
                flag = " ← baseline" if abs(frac-base_frac)<1e-6 else ""
                print(f"    {frac:6.2f}  {f_y:5d}  {s_y:5d}  {fw_cm:6.1f}  {sd_cm:6.1f}  {circ:8.2f}  {error:+7.2f}{flag}")
                if abs(error)<best_err:
                    best_err=abs(error); best_row={"frac":frac,"f_y":f_y,"s_y":s_y,"fw_cm":fw_cm,"sd_cm":sd_cm,"circ_cm":circ,"error":error}
                sweep_results.append({
                    "participant_id":pid,"level":level_name,"frac":frac,
                    "is_baseline":abs(frac-base_frac)<1e-6,
                    "f_y":f_y,"s_y":s_y,"fw_cm":fw_cm,"sd_cm":sd_cm,"circ_cm":circ,"error":error})
            if best_row:
                print(f"    → Best: frac={best_row['frac']:.2f}  circ={best_row['circ_cm']:.2f}cm  error={best_row['error']:+.2f}cm")
                if abs(best_row["frac"]-base_frac)>0.01:
                    print(f"    ⚠  Best fraction differs from baseline by {abs(best_row['frac']-base_frac)*100:.0f}% of torso span")
                else:
                    print(f"    ✓  Baseline fraction near-optimal for {pid}")
            pid_rows = [r for r in sweep_results if r["participant_id"]==pid and r["level"]==level_name]
            if pid_rows:
                circs = [r["circ_cm"] for r in pid_rows]
                print(f"    Sensitivity: {min(circs):.2f}–{max(circs):.2f} cm "
                      f"(range={max(circs)-min(circs):.2f}cm over ±12% span)")
    return sweep_results

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("FitLoop Model 1 — Stage 7D Diagnostic Audit")
    sessions = load_ground_truth_csv(str(CSV_PATH))
    if not sessions: print("ERROR: no sessions loaded"); sys.exit(1)
    pose_ex   = PoseExtractor()
    seg_ex    = BodySegmenter()
    feat_ex   = GeometricFeatureExtractor()
    try:
        gt      = audit_ground_truth(sessions)
        calib   = audit_calibration(sessions, pose_ex, seg_ex, feat_ex)
        slices  = audit_anatomical_slices(sessions, pose_ex, seg_ex, feat_ex)
        audit_error_hypotheses(sessions, slices, calib)
        sweep   = audit_vertical_sweep(sessions, pose_ex, seg_ex, feat_ex)
    finally:
        if hasattr(pose_ex,"close"): pose_ex.close()
        if hasattr(seg_ex,"close"):  seg_ex.close()
    out = OUT_DIR/"stage7d_results.json"
    with open(out,"w") as f:
        json.dump({"ground_truth_audit":gt,"calibration_table":calib,
                   "slice_table":slices,"sweep_results":sweep},
                  f, indent=2, default=str)
    print(f"\n✓ Results: {out}")
    print(f"  Overlays: {OUT_DIR}/")

if __name__=="__main__":
    main()
