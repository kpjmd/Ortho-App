# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Frontend (React)
- **Development**: `cd frontend && npm start` - Starts React dev server on port 3000
- **Build**: `cd frontend && npm run build` - Creates production build
- **Test**: `cd frontend && npm test` - Runs Jest tests
- **Lint**: `cd frontend && npx eslint src/` - Runs ESLint (configured in package.json)

### Backend (FastAPI)
- **Development**: `cd backend && uvicorn server:app --reload --port 8001` - Starts FastAPI server on port 8001
- **Test**: `cd backend && pytest` - Runs pytest tests
- **Lint**: `cd backend && flake8 .` - Runs flake8 linting
- **Format**: `cd backend && black .` - Formats code with black
- **Type Check**: `cd backend && mypy .` - Runs mypy type checking

### Docker
- **Build**: `docker build -t ortho-app .` - Builds multi-stage Docker image
- **Run**: `docker run -p 80:80 ortho-app` - Runs containerized app

## Architecture Overview

This is a full-stack orthopedic recovery tracking application with the following structure:

### Backend (FastAPI)
- **Core**: `backend/server.py` - Main FastAPI application with all API endpoints
- **Database**: MongoDB with Motor (async driver)
- **Models**: Pydantic models for Patient, WearableData, Survey, AIInsight
- **AI Logic**: Simple rule-based insight generation in `generate_insights()` function

### Frontend (React)
- **Main App**: `frontend/src/App.js` - Single-file React app with all components
- **Styling**: Tailwind CSS for styling
- **Routing**: React Router for navigation
- **API**: Axios for backend communication

### Key Features
- **Patient Management**: Support for 10 common orthopedic diagnoses (5 knee, 5 shoulder)
- **Validated PRO Questionnaires**: KOOS (knee injuries) and ASES (shoulder injuries) with clinical scoring
- **Survey Collection**: Basic injury surveys with pain, mobility, ROM, strength, and ADL metrics
- **Wearable Data**: Track steps, heart rate, oxygen saturation, sleep, walking speed
- **AI Insights**: Enhanced recovery analysis using validated PRO scores

### Supported Diagnoses
- **Knee Diagnoses (use KOOS)**:
  - ACL Tear, Meniscus Tear, Cartilage Defect, Knee Osteoarthritis, Post Total Knee Replacement
- **Shoulder Diagnoses (use ASES)**:
  - Rotator Cuff Tear, Labral Tear, Shoulder Instability, Shoulder Osteoarthritis, Post Total Shoulder Replacement

### Data Models
- **Patient**: Basic info, diagnosis_type (DiagnosisType enum), body_part property (auto-determined)
- **KOOSResponse/KOOSScores**: KOOS questionnaire (42 items) with subscale scoring for knee patients
- **ASESResponse/ASESScores**: ASES questionnaire (pain + 10 function items) for shoulder patients
- **Survey**: Basic pain/mobility scores, range of motion, strength, activities of daily living
- **WearableData**: Daily metrics from wearable devices
- **AIInsight**: Recovery status using PRO scores with clinical recommendations

### PRO Questionnaire API Endpoints
- **KOOS (knee patients)**:
  - `POST /api/koos` - Submit KOOS questionnaire (42 items, 0-4 scale)
  - `GET /api/koos/{patient_id}` - Get KOOS score history
  - `GET /api/koos/{patient_id}/latest` - Get latest KOOS scores
  - `GET /api/koos/{patient_id}/trends` - Get score trends over time
- **ASES (shoulder patients)**:
  - `POST /api/ases` - Submit ASES questionnaire (pain VAS + 10 function items)
  - `GET /api/ases/{patient_id}` - Get ASES score history
  - `GET /api/ases/{patient_id}/latest` - Get latest ASES scores
  - `GET /api/ases/{patient_id}/trends` - Get score trends over time

### Standard Workflow

First think through the problem, read the codebase for relevant files, and write a plan to TODO.md.
The plan should have a list of todo items that you can check off as you complete them
Before you begin working, check in with me and I will verify the plan.
Then, begin working on the todo items, marking them as complete as you go.
Please every step of the way just give me a high level explanation of what changes you made
Make every task and code change you do as simple as possible. We want to avoid making any massive or complex changes. Every change should impact as little code as possible. Everything is about simplicity.
Finally, add a review section to the TODO.md file with a summary of the changes you made and any other relevant information.

### Development Notes
- Frontend and backend communicate via REST API at `/api` prefix
- MongoDB collections: `patients`, `surveys`, `wearable_data`, `ai_insights`
- Sample data can be generated via `/api/sample-data` endpoint
- Injury-specific forms adapt based on ACL vs Rotator Cuff selection
- Docker deployment uses nginx proxy with both services