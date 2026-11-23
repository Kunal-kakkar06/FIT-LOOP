import sys, math
sys.stdout.reconfigure(encoding='utf-8')

# Ensure wardrobe_ml is in sys.path
sys.path.insert(0, r"C:\Users\Lenovo\Downloads\Fitloop\wardrobe_ml")
sys.path.insert(0, r"C:\Users\Lenovo\Downloads\Fitloop\wardrobe_ml\src")

from annotation_converter import DeepFashion2ToYOLOConverter

conv = DeepFashion2ToYOLOConverter()
w, h = 500, 600

print("=== RUNNING CONVERTER UNIT TESTS ===")

# 1. Valid box test
valid_bbox = [100, 150, 300, 450]
ok, norm_box, reason, details = conv.validate_and_normalize_box(valid_bbox, w, h)
assert ok, f"Failed on valid box: {reason}, {details}"
xc, yc, bw, bh = norm_box
expected_xc = ((100 + 300) / 2.0) / 500.0  # 200 / 500 = 0.40
expected_yc = ((150 + 450) / 2.0) / 600.0  # 300 / 600 = 0.50
expected_bw = (300 - 100) / 500.0          # 200 / 500 = 0.40
expected_bh = (450 - 150) / 600.0          # 300 / 600 = 0.50
assert math.isclose(xc, expected_xc), f"xc mismatch: {xc} vs {expected_xc}"
assert math.isclose(yc, expected_yc), f"yc mismatch: {yc} vs {expected_yc}"
assert math.isclose(bw, expected_bw), f"bw mismatch: {bw} vs {expected_bw}"
assert math.isclose(bh, expected_bh), f"bh mismatch: {bh} vs {expected_bh}"
print("[PASS] Valid bounding box normalization matches exact formula")

# 2. Known degenerate box [0, 0, 0, 0]
ok, norm_box, reason, details = conv.validate_and_normalize_box([0, 0, 0, 0], w, h)
assert not ok, "Known degenerate [0,0,0,0] was not rejected!"
assert reason == "zero_or_negative_area", f"Wrong reason for [0,0,0,0]: {reason}"
print(f"[PASS] Known degenerate [0, 0, 0, 0] correctly rejected: {reason}")

# 3. Negative area box
ok, norm_box, reason, details = conv.validate_and_normalize_box([200, 300, 100, 150], w, h)
assert not ok and reason == "zero_or_negative_area"
print(f"[PASS] Negative area box correctly rejected: {reason}")

# 4. Zero width box
ok, norm_box, reason, details = conv.validate_and_normalize_box([100, 50, 100, 250], w, h)
assert not ok and reason == "zero_or_negative_area"
print(f"[PASS] Zero width box correctly rejected: {reason}")

# 5. Out of bounds coordinates (negative)
ok, norm_box, reason, details = conv.validate_and_normalize_box([-5, 50, 100, 250], w, h)
assert not ok and reason == "out_of_bounds_coordinates"
print(f"[PASS] Negative out-of-bounds correctly rejected: {reason}")

# 6. Out of bounds coordinates (exceeds width)
ok, norm_box, reason, details = conv.validate_and_normalize_box([100, 50, 505, 250], w, h)
assert not ok and reason == "out_of_bounds_coordinates"
print(f"[PASS] Exceeds-width out-of-bounds correctly rejected: {reason}")

# 7. Non-numeric coordinates
ok, norm_box, reason, details = conv.validate_and_normalize_box([100, "invalid", 200, 250], w, h)
assert not ok and reason == "non_numeric_coordinates"
print(f"[PASS] Non-numeric coordinate correctly rejected: {reason}")

# 8. Non-finite coordinates (NaN)
ok, norm_box, reason, details = conv.validate_and_normalize_box([100, float('nan'), 200, 250], w, h)
assert not ok and reason == "non_finite_coordinates"
print(f"[PASS] NaN coordinate correctly rejected: {reason}")

# 9. Non-finite coordinates (Inf)
ok, norm_box, reason, details = conv.validate_and_normalize_box([100, 50, float('inf'), 250], w, h)
assert not ok and reason == "non_finite_coordinates"
print(f"[PASS] Inf coordinate correctly rejected: {reason}")

# 10. Malformed bounding box
ok, norm_box, reason, details = conv.validate_and_normalize_box([100, 50, 200], w, h)
assert not ok and reason == "malformed_bounding_box"
print(f"[PASS] Length 3 bounding box correctly rejected: {reason}")

# 11. Category validation
assert conv.validate_category(1) == (True, 0, "")
assert conv.validate_category(13) == (True, 12, "")
assert conv.validate_category(0)[0] is False  # DF2 is 1-indexed
assert conv.validate_category(14)[0] is False # DF2 only has 1..13
assert conv.validate_category(None)[0] is False
assert conv.validate_category(True)[0] is False # bool disallowed
print("[PASS] Category validation tests passed (1..13 -> 0..12, others rejected)")

print("\nALL 11 UNIT TESTS PASSED SUCCESSFULLY!")
