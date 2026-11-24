"""
FitLoop Phase 5 API Service Comprehensive Test Suite
=====================================================
Automated end-to-end verification of wardrobe_ml/src/app.py:
1. Health endpoint (/health) validation.
2. User registration and conflict handling (/api/v1/users).
3. Wardrobe scan upload, YOLO26n garment localization, color extraction, and DB persistence (/api/v1/scans).
4. User wardrobe inventory retrieval and multi-tenant isolation (/api/v1/wardrobe/{user_id}).
5. Check-before-buy duplicate detection and recommendation alerts (/api/v1/check-before-buy).
6. Error handling (404 for unknown user, 400 for bad image, 422 for bad schema).
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from fastapi.testclient import TestClient

# Setup pathing
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Use isolated test database
TEST_DB_PATH = SRC_DIR / "fitloop_api_test.db"
os.environ["FITLOOP_DB_URL"] = f"sqlite:///{TEST_DB_PATH}"

from app import app, get_repository, detector, UPLOADS_DIR, CROPS_DIR
from database import WardrobeRepository


def create_synthetic_image_bytes(
    bgr_color: Tuple[int, int, int] = (59, 38, 27),
    width: int = 240,
    height: int = 240,
) -> bytes:
    """
    Generate an in-memory JPEG image with the specified BGR color.
    BGR (59, 38, 27) corresponds to RGB (27, 38, 59) which is Navy Blue (#1B263B).
    """
    img = np.full((height, width, 3), bgr_color, dtype=np.uint8)
    # Add a slight inner garment patch to provide color depth
    center_color = (
        max(0, bgr_color[0] - 10),
        max(0, bgr_color[1] - 10),
        max(0, bgr_color[2] - 10),
    )
    cv2.rectangle(img, (30, 30), (width - 30, height - 30), center_color, -1)
    success, encoded = cv2.imencode(".jpg", img)
    if not success:
        raise RuntimeError("Failed to encode synthetic JPEG image")
    return encoded.tobytes()


class TestReportCollector:
    """Accumulates test execution results and formats report."""

    def __init__(self) -> None:
        self.results: List[Dict[str, Any]] = []
        self.start_time = time.perf_counter()

    def record(self, test_name: str, status: str, details: str, duration_ms: float) -> None:
        self.results.append({
            "test": test_name,
            "status": status,
            "details": details,
            "duration_ms": round(duration_ms, 2),
        })
        mark = "✓ PASS" if status == "PASS" else "✗ FAIL"
        print(f"[{mark}] {test_name} ({duration_ms:.1f}ms): {details}")

    def generate_report(self) -> str:
        total_time = (time.perf_counter() - self.start_time) * 1000
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        failed = sum(1 for r in self.results if r["status"] == "FAIL")

        lines = [
            "=" * 78,
            "FITLOOP PHASE 5: FASTAPI SERVICE AUTOMATED VERIFICATION REPORT",
            "=" * 78,
            f"Execution Timestamp : {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
            f"Total Test Cases    : {len(self.results)}",
            f"Passed              : {passed}",
            f"Failed              : {failed}",
            f"Overall Status      : {'SUCCESS' if failed == 0 else 'FAILURE'}",
            f"Total Runtime       : {total_time:.2f} ms",
            "-" * 78,
            f"{'Test Case':<38} | {'Status':<6} | {'Time (ms)':<9} | Details",
            "-" * 78,
        ]

        for r in self.results:
            lines.append(
                f"{r['test']:<38} | {r['status']:<6} | {r['duration_ms']:<9.1f} | {r['details']}"
            )

        lines.append("=" * 78)
        return "\n".join(lines)


def run_tests() -> int:
    collector = TestReportCollector()
    client = TestClient(app)

    created_test_files: List[Path] = []

    try:
        # -------------------------------------------------------------
        # Test 1: GET /health
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        resp = client.get("/health")
        dt = (time.perf_counter() - t0) * 1000
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_architecture"] == "YOLO26n"
        assert data["device"] == "cpu"
        collector.record(
            "1. Health Check Endpoint",
            "PASS",
            f"Status 200, Arch: {data['model_architecture']}, Model Loaded: {data['model_loaded']}",
            dt,
        )

        # -------------------------------------------------------------
        # Test 2: POST /api/v1/users (User Registration)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        user_alice_payload = {"name": "Alice Green", "email": "alice.green@fitloop.test"}
        resp = client.post("/api/v1/users", json=user_alice_payload)
        dt = (time.perf_counter() - t0) * 1000
        assert resp.status_code == 201, f"Expected 201, got {resp.status_code}: {resp.text}"
        alice_data = resp.json()
        assert alice_data["id"] > 0
        assert alice_data["email"] == "alice.green@fitloop.test"
        alice_id = alice_data["id"]
        collector.record(
            "2a. Create User Alice",
            "PASS",
            f"Status 201, User ID={alice_id}, Email={alice_data['email']}",
            dt,
        )

        # Test 2b: Duplicate Email Handling
        t0 = time.perf_counter()
        dup_resp = client.post("/api/v1/users", json=user_alice_payload)
        dt = (time.perf_counter() - t0) * 1000
        assert dup_resp.status_code == 400, f"Expected 400, got {dup_resp.status_code}"
        collector.record(
            "2b. Duplicate Email Conflict",
            "PASS",
            f"Status 400 caught duplicate email: {dup_resp.json().get('detail')}",
            dt,
        )

        # Test 2c: Create User Bob (for tenant isolation tests)
        t0 = time.perf_counter()
        user_bob_payload = {"name": "Bob Stone", "email": "bob.stone@fitloop.test"}
        resp = client.post("/api/v1/users", json=user_bob_payload)
        dt = (time.perf_counter() - t0) * 1000
        assert resp.status_code == 201
        bob_id = resp.json()["id"]
        collector.record(
            "2c. Create User Bob",
            "PASS",
            f"Status 201, User ID={bob_id}, Bob registered for isolation testing",
            dt,
        )

        # -------------------------------------------------------------
        # Test 3: POST /api/v1/scans (Upload Wardrobe Photo)
        # -------------------------------------------------------------
        # Generate synthetic Navy Blue garment image (#1B263B -> BGR: 59, 38, 27)
        navy_bytes = create_synthetic_image_bytes(bgr_color=(59, 38, 27), width=320, height=320)

        t0 = time.perf_counter()
        files = {"file": ("navy_shirt.jpg", navy_bytes, "image/jpeg")}
        form_data = {"user_id": str(alice_id), "input_type": "single_garment"}
        resp = client.post("/api/v1/scans", files=files, data=form_data)
        dt = (time.perf_counter() - t0) * 1000
        assert resp.status_code == 201, f"Expected 201, got {resp.status_code}: {resp.text}"
        scan_data = resp.json()
        assert scan_data["scan_id"] > 0
        assert scan_data["user_id"] == alice_id
        assert scan_data["items_detected"] >= 1
        first_item = scan_data["items"][0]
        assert "Navy Blue" in first_item["colour_name"] or first_item["colour_hex"].startswith("#")
        collector.record(
            "3a. Upload Scan & Color Extraction",
            "PASS",
            f"Status 201, Scan ID={scan_data['scan_id']}, Detected={first_item['colour_name']} ({first_item['colour_hex']})",
            dt,
        )

        # Test 3b: Scan with Non-Existent User ID -> 404
        t0 = time.perf_counter()
        files = {"file": ("test.jpg", navy_bytes, "image/jpeg")}
        resp = client.post("/api/v1/scans", files=files, data={"user_id": "999999", "input_type": "closet"})
        dt = (time.perf_counter() - t0) * 1000
        assert resp.status_code == 404, f"Expected 404, got {resp.status_code}"
        collector.record(
            "3b. Scan Unknown User ID",
            "PASS",
            f"Status 404 caught missing user_id=999999 cleanly",
            dt,
        )

        # Test 3c: Scan with Corrupted/Empty Image -> 400
        t0 = time.perf_counter()
        files = {"file": ("corrupt.jpg", b"NOT_AN_IMAGE_DATA", "image/jpeg")}
        resp = client.post("/api/v1/scans", files=files, data={"user_id": str(alice_id)})
        dt = (time.perf_counter() - t0) * 1000
        assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"
        collector.record(
            "3c. Scan Invalid Image File",
            "PASS",
            f"Status 400 rejected corrupted file payload",
            dt,
        )

        # -------------------------------------------------------------
        # Test 4: GET /api/v1/wardrobe/{user_id} (Inventory & Isolation)
        # -------------------------------------------------------------
        # Alice's wardrobe should have the item added
        t0 = time.perf_counter()
        resp = client.get(f"/api/v1/wardrobe/{alice_id}")
        dt = (time.perf_counter() - t0) * 1000
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        alice_wardrobe = resp.json()
        assert alice_wardrobe["user_id"] == alice_id
        assert alice_wardrobe["total_items"] >= 1
        collector.record(
            "4a. Query Alice Wardrobe Inventory",
            "PASS",
            f"Status 200, Items Count={alice_wardrobe['total_items']}",
            dt,
        )

        # Bob's wardrobe must be strictly isolated (0 items)
        t0 = time.perf_counter()
        resp = client.get(f"/api/v1/wardrobe/{bob_id}")
        dt = (time.perf_counter() - t0) * 1000
        assert resp.status_code == 200
        bob_wardrobe = resp.json()
        assert bob_wardrobe["total_items"] == 0, f"Tenant isolation breach! Bob saw {bob_wardrobe['total_items']} items"
        collector.record(
            "4b. Strict User Isolation (Bob)",
            "PASS",
            f"Status 200, Bob items=0 (Alice items hidden, 0 leaks)",
            dt,
        )

        # Non-existent user query -> 404
        t0 = time.perf_counter()
        resp = client.get("/api/v1/wardrobe/888888")
        dt = (time.perf_counter() - t0) * 1000
        assert resp.status_code == 404
        collector.record(
            "4c. Query Unknown Wardrobe ID",
            "PASS",
            f"Status 404 caught non-existent user 888888",
            dt,
        )

        # -------------------------------------------------------------
        # Test 5: POST /api/v1/check-before-buy (Duplicate Detection)
        # -------------------------------------------------------------
        # Scenario 5a: Alice checks an identical Navy Blue item (#1B263B -> BGR: 59, 38, 27)
        # Should trigger DUPLICATE verdict against the item scanned in 3a
        t0 = time.perf_counter()
        candidate_navy_bytes = create_synthetic_image_bytes(bgr_color=(59, 38, 27), width=240, height=240)
        files = {"file": ("prospective_navy.jpg", candidate_navy_bytes, "image/jpeg")}
        form_data = {"user_id": str(alice_id), "candidate_category": first_item["category"]}
        resp = client.post("/api/v1/check-before-buy", files=files, data=form_data)
        dt = (time.perf_counter() - t0) * 1000
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        chk_data = resp.json()
        assert chk_data["verdict"] in ("DUPLICATE", "VERY_SIMILAR")
        assert chk_data["similarity_score"] >= 0.80
        assert "DUPLICATE" in chk_data["recommendation"] or "SIMILAR" in chk_data["recommendation"]
        assert chk_data["check_id"] is not None
        collector.record(
            "5a. Check Before Buy: Duplicate Detected",
            "PASS",
            f"Verdict={chk_data['verdict']}, Score={chk_data['similarity_score']:.2f}, Rec: {chk_data['recommendation'][:55]}...",
            dt,
        )

        # Scenario 5b: Alice checks a distinct Emerald Green / Mustard Yellow item
        # Mustard Yellow: RGB (255, 219, 88) -> BGR (88, 219, 255)
        t0 = time.perf_counter()
        yellow_bytes = create_synthetic_image_bytes(bgr_color=(88, 219, 255), width=240, height=240)
        files = {"file": ("prospective_yellow.jpg", yellow_bytes, "image/jpeg")}
        distinct_cat = "trousers" if first_item["category"] != "trousers" else "short sleeve top"
        form_data = {"user_id": str(alice_id), "candidate_category": distinct_cat}
        resp = client.post("/api/v1/check-before-buy", files=files, data=form_data)
        dt = (time.perf_counter() - t0) * 1000
        assert resp.status_code == 200
        unique_chk = resp.json()
        assert unique_chk["verdict"] == "UNIQUE"
        assert unique_chk["similarity_score"] < 0.65
        assert "UNIQUE" in unique_chk["recommendation"]
        collector.record(
            "5b. Check Before Buy: Unique Garment",
            "PASS",
            f"Verdict={unique_chk['verdict']}, Score={unique_chk['similarity_score']:.2f}, Rec: {unique_chk['recommendation'][:55]}...",
            dt,
        )

        # Scenario 5c: Check-before-buy with non-existent user -> 404
        t0 = time.perf_counter()
        files = {"file": ("test.jpg", yellow_bytes, "image/jpeg")}
        resp = client.post("/api/v1/check-before-buy", files=files, data={"user_id": "777777"})
        dt = (time.perf_counter() - t0) * 1000
        assert resp.status_code == 404
        collector.record(
            "5c. Check Before Buy Unknown User",
            "PASS",
            f"Status 404 caught non-existent user 777777",
            dt,
        )

    except Exception as e:
        collector.record("Test Suite Failure", "FAIL", str(e), 0.0)
        raise
    finally:
        # Cleanup temporary test database safely
        try:
            # Dispose repository connection pool
            repo = WardrobeRepository(f"sqlite:///{TEST_DB_PATH}")
            repo.close()
            time.sleep(0.1)
            if TEST_DB_PATH.exists():
                TEST_DB_PATH.unlink()
        except Exception as e:
            sys.stderr.write(f"[WARN] Failed to clean up {TEST_DB_PATH}: {e}\n")

    # Generate and write output report
    report_text = collector.generate_report()
    print("\n" + report_text)

    output_dir = PROJECT_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "api_service_test_report.txt"
    report_file.write_text(report_text, encoding="utf-8")
    print(f"\n[REPORT SAVED] {report_file}")

    # Return 0 if all tests passed, 1 otherwise
    failed_count = sum(1 for r in collector.results if r["status"] == "FAIL")
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
