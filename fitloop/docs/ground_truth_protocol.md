# FitLoop Model 1 — Ground-Truth Evaluation Protocol

**Protocol Version:** 1.0  
**Target:** Validation of Model 1 2D Calibrated Circumference Estimator against Anthropometric Tape Measurements.

---

## 1. Ethical & Privacy Protections
1. **Informed Consent:** Every participant must sign an explicit research consent form authorizing manual anthropometric tape measurements and photograph capture.
2. **Pseudonymization:** Real identities are never stored in datasets. Participants are assigned an arbitrary pseudonymous code (`P001`, `P002`, ...).
3. **Data Segregation:** 
   - Tabular measurements are stored in `data/ground_truth/`.
   - Raw photograph files are stored in `data/raw/` and are strictly excluded from source control via `.gitignore`.
   - No biometric images or records are uploaded to external cloud services or repositories.

---

## 2. Standardized Capture Environment & Suggestions
> [!NOTE]
> Camera distance and height figures ($2.8\text{ m}$ distance, $1.0\text{ m}$ height) are practical capture suggestions, not universal anthropometric standards. They must be validated against the specific camera optical field-of-view (FOV) and physical room dimensions in use.

- **Recommended Setup Suggestions:**
  - **Camera Configuration:** Rigid tripod mounting at approximately mid-torso height ($\sim 1.0\text{ m}$). Camera optical axis level ($0^\circ$ pitch, $0^\circ$ roll).
  - **Distance to Subject:** Approximately $2.5\text{--}3.0\text{ m}$ to ensure full-body unclipped framing while minimizing perspective convergence.
  - **Subject Turntable / Position Marker:** Fixed cross taped to the floor. Subject must stand on the identical marker for front view and pivot $90^\circ$ in place for side view.
- **Subject Stance & Posture:**
  - **Front View:** Upright neutral A-pose. Arms abducted $20^\circ\text{--}30^\circ$ from torso; feet shoulder-width apart; looking forward (Frankfurt plane horizontal).
  - **Side View:** Upright $90^\circ$ profile. Arms held slightly forward ($15^\circ$) or resting across clavicles so the lateral and sagittal torso contours remain completely unoccluded.
- **Clothing & Framing Guidelines:**
  - **Clothing:** Form-fitting athletic compression garments (tucked in). Hair tied tightly on top of head. Barefoot or thin socks.
  - **Vertical Framing:** Leave at least $5\%$ (recommended $10\%$) headroom above the head apex and floor space below the heels.

---

## 2.1 Automated Capture Quality Gate (Stage 7G)
Every paired submission is evaluated by an automated software gate prior to circumference inference:
1. **Vertical Margin Threshold:** Minimum $5.0\%$ frame headroom and foot margin (provisional default). Touching frame edge ($\le 2\text{ px}$) triggers immediate `REJECT`.
2. **Scale Discrepancy Threshold:** Maximum cross-view scale difference $\le 5.0\%$ (provisional strict threshold). Discrepancy $> 5.0\%$ triggers `WARNING`; $> 15.0\%$ triggers `REJECT`.
3. **Geometry Consistency:** Front and side captures must match in dimensions and aspect ratio ($\le 2\%$ tolerance).
4. **Notice:** A quality gate `PASS` indicates compliance with basic geometric capture criteria; it does NOT establish anthropometric accuracy or clinical validity.

## 3. Anthropometric Reference Measurement Procedure (ISO 7250 / ISO 8559)
Measurements are taken using a non-stretchable medical anthropometric tape measure with spring-loaded tension control:

1. **Standing Height (Stature):**
   - Measured with stadiometer or vertical wall rule without shoes. Subject stands erect with head in the Frankfurt horizontal plane.
2. **Chest Circumference:**
   - Tape placed horizontally at the level of maximum chest circumference (axillary fold level / mid-sternum), measured at the end of normal expiration.
3. **Waist Circumference:**
   - Tape placed horizontally at the midway point between the inferior margin of the ribcage and the top of the iliac crest (natural waist indentation).
4. **Abdomen Circumference:**
   - Tape placed horizontally across the maximum anterior protrusion of the abdomen (umbilicus level).
5. **Hips Circumference:**
   - Tape placed horizontally around the maximum posterior protrusion of the buttocks and greater trochanters.

### Protocol for Repeated Measurements & Repeatability
- Two independent readings are collected for each dimension by a trained operator.
- If $|R_1 - R_2| \le 0.5\text{ cm}$, the reference value is the arithmetic mean:
  $$\text{Ref} = \frac{R_1 + R_2}{2}$$
- If $|R_1 - R_2| > 0.5\text{ cm}$, a third reading $R_3$ is immediately taken, and the reference value is the 3-reading mean.
- The inter-reading difference is explicitly logged to quantify manual measurement uncertainty.

---

## 4. Participant-Level Dataset Partitioning
To prevent **participant leakage** across training/calibration and evaluation splits:
- Dataset partitioning is performed **strictly by participant ID**, not by image or session.
- All sessions of a participant (e.g. both fitted and standard clothing) belong exclusively to either:
  - **Calibration Set (40%):** Used for investigating systematic bias or potential future parameter tuning.
  - **Evaluation Set (60%):** Held-out test set evaluated strictly once for empirical validation.

---

## 5. Metrics & Success Criteria
For each dimension independently:
- **Mean Absolute Error (MAE):** $\text{MAE} = \frac{1}{N_{\text{valid}}} \sum |y - \hat{y}|$
- **Root Mean Squared Error (RMSE):** $\text{RMSE} = \sqrt{\frac{1}{N_{\text{valid}}} \sum (y - \hat{y})^2}$
- **Mean Signed Bias:** $\text{Bias} = \frac{1}{N_{\text{valid}}} \sum (\hat{y} - y)$
- **Precision Thresholds:** Percentage within $\pm 1.0\text{ cm}$ and $\pm 2.0\text{ cm}$.
- **Coverage:** $\frac{N_{\text{valid}}}{N_{\text{total}}}$. Failed detections are reported explicitly, never ignored.
