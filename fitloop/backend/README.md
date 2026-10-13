# Backend API Service

Owner: Monish

FastAPI backend application orchestrating:
- User authentication and scan session persistence.
- Image storage and quality gating orchestration.
- Interfacing with Model 1 (Body Measurement Engine) and Model 2/3 (Garment Fit & Sizing).
- Exposing REST API endpoints for the FitLoop frontend client.

## Structure
- `app/api/`: REST route endpoints (`/scans`, `/measurements`, `/recommendations`).
- `app/db/`: Database models, migrations, and session management.
- `app/schemas/`: Pydantic request and response schemas.
- `app/services/`: Business logic services, including model inference wrappers.
- `tests/`: Automated unit and integration tests.
