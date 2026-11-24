import cv2
import numpy as np

def main():
    from fitloop_measurement.pose import PoseExtractor
    from fitloop_measurement.segmentation import BodySegmenter
    from fitloop_measurement.features import GeometricFeatureExtractor
    from fitloop_measurement.measurements.slice_selector import select_anatomical_slice_from_mask
    from fitloop_measurement.measurements.schemas import AnatomicalLevel

    front_path = "/Users/kunalkakkar/Desktop/WhatsApp Image 2026-10-14 at 20.12.06.jpeg"
    front_img = cv2.imread(front_path)

    pose_ext = PoseExtractor()
    seg = BodySegmenter()
    feat_ext = GeometricFeatureExtractor()

    p_res = pose_ext.extract(front_img, "front")
    s_res = seg.segment(front_img, "front")
    f_res = feat_ext.extract_view_features(p_res, s_res)

    for lvl in [AnatomicalLevel.CHEST, AnatomicalLevel.WAIST, AnatomicalLevel.ABDOMEN, AnatomicalLevel.HIPS]:
        slc, res = select_anatomical_slice_from_mask(lvl, f_res, s_res.mask)
        print(f"{lvl.value.upper()}: y={slc.image_y if slc else None}, width_px={slc.width_pixels if slc else None}, valid={res.is_valid}, warnings={res.warnings}")

if __name__ == '__main__':
    main()
