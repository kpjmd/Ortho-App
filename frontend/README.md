# RcvryAI - Orthopedic Recovery Monitoring Platform

**OrthoTrack** is a comprehensive, clinical-grade orthopedic recovery monitoring platform that empowers patients, healthcare providers, and researchers with validated PRO questionnaires, wearable data integration, and ML-powered analytics for optimized recovery outcomes.

## 🎯 Vision & Purpose

RcvryAI transforms orthopedic care through:
- **Evidence-based recovery tracking** using validated clinical questionnaires (KOOS, ASES)
- **Real-time wearable data integration** for comprehensive patient monitoring
- **Predictive analytics** to identify recovery risks and optimize treatment plans
- **Clinical-grade data validation** ensuring research and healthcare compliance

### Target Users
- **Patients**: Track recovery progress with validated clinical tools
- **Healthcare Providers**: Monitor patient outcomes and identify intervention opportunities  
- **Researchers**: Access validated datasets for orthopedic recovery studies

---

## 🏗️ Technology Stack & Architecture

### Frontend
- **React 19** with modern hooks and functional components
- **Tailwind CSS** for responsive, healthcare-focused UI design
- **Recharts** for advanced data visualization and analytics
- **React Router** for seamless navigation
- **Axios** for robust API communication

### Backend  
- **FastAPI** with async/await for high-performance API endpoints
- **MongoDB** with Motor async driver for clinical data storage
- **Pydantic** for data validation and serialization
- **PyJWT** for secure authentication

### Analytics & ML
- **Recovery Trajectory Models** for personalized outcome prediction
- **Clinical Alert System** for real-time risk identification
- **Correlation Analysis Engine** linking wearable data to recovery outcomes
- **Quality Monitoring** ensuring data integrity and clinical compliance

### Deployment
- **Docker** multi-stage build for production deployment
- **Nginx** reverse proxy for frontend/backend integration
- **Environment-based configuration** for development/staging/production

---

## 🚀 Key Features

### Patient Management
- **10 Supported Diagnoses**: 5 knee conditions (ACL Tear, Meniscus Tear, Cartilage Defect, Knee Osteoarthritis, Post Total Knee Replacement) and 5 shoulder conditions (Rotator Cuff Tear, Labral Tear, Shoulder Instability, Shoulder Osteoarthritis, Post Total Shoulder Replacement)
- **Diagnosis-specific workflows** with targeted questionnaires and recovery trajectories

### Validated PRO Questionnaires
- **KOOS (Knee injury and Osteoarthritis Outcome Score)**: 42-item questionnaire with 5 subscales for knee patients
- **ASES (American Shoulder and Elbow Surgeons)**: Pain VAS + 10 function items for shoulder patients
- **Clinical scoring algorithms** providing standardized outcome measures
- **Trend analysis** tracking recovery progress over time

### Wearable Data Integration
- **Comprehensive metrics**: Steps, heart rate, oxygen saturation, sleep quality, walking speed
- **Quality monitoring** with data validation and missing data handling
- **Bulk import capabilities** supporting multiple wearable device formats
- **Real-time synchronization** for continuous monitoring

### Advanced Analytics Dashboard (17+ Components)
- **Recovery Velocity Chart**: Tracks pace of improvement relative to clinical benchmarks
- **Predictive Insights Dashboard**: ML-powered recovery outcome predictions
- **Clinical Alerts Panel**: Real-time risk identification and intervention recommendations
- **Cardiovascular Recovery Monitor**: Heart rate variability and fitness tracking
- **Sleep Recovery Analysis**: Sleep quality correlation with recovery outcomes
- **Correlation Analysis View**: Multi-variate analysis of recovery factors
- **Provider Dashboard**: Clinical overview for healthcare professionals
- **Risk Indicators**: Early warning system for recovery complications

### Data Management
- **Manual Data Entry**: Healthcare provider interface for clinical data input
- **Bulk Data Import**: CSV/JSON import from wearable devices and EMR systems
- **Data Quality Monitoring**: Automated validation and quality scoring
- **Audit Trail**: Complete change tracking for clinical compliance

---

## 📋 Prerequisites

### Development Environment
- **Node.js**: Version 20.x or higher
- **Python**: Version 3.11 or higher  
- **MongoDB**: Version 6.0 or higher
- **npm/yarn**: Latest stable version

### Optional Tools
- **Docker**: For containerized development and deployment
- **Git**: Version control and collaboration

---

## 🛠️ Setup Instructions

### Backend Setup

1. **Navigate to backend directory**
   ```bash
   cd backend
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration:
   # MONGO_URL=mongodb://localhost:27017
   # DB_NAME=ortho_recovery
   # JWT_SECRET=your-secret-key
   # API_BASE_URL=http://localhost:8001
   ```

4. **Start FastAPI development server**
   ```bash
   uvicorn server:app --reload --host 0.0.0.0 --port 8001
   ```

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies**
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Environment configuration**
   ```bash
   # Create .env file in frontend directory
   echo "REACT_APP_BACKEND_URL=http://localhost:8001" > .env
   ```

4. **Start React development server**
   ```bash
   npm start
   # or
   yarn start
   ```

5. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8001
   - API Documentation: http://localhost:8001/docs

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t rcvryai-orthotrack .
   ```

2. **Run the containerized application**
   ```bash
   docker run -p 80:80 \
     -e MONGO_URL=mongodb://your-mongo-host:27017 \
     -e DB_NAME=ortho_recovery \
     rcvryai-orthotrack
   ```

---

## 💻 Development Workflow

### Available Scripts

#### Frontend Commands
```bash
# Development server
npm start                    # Starts React dev server on port 3000

# Production build
npm run build               # Creates optimized production build

# Testing
npm test                    # Runs Jest test suite in watch mode

# Code quality
npx eslint src/             # Runs ESLint code analysis
```

#### Backend Commands
```bash
# Development server
uvicorn server:app --reload --port 8001    # Starts FastAPI with auto-reload

# Testing
pytest                      # Runs pytest test suite
pytest --cov               # Runs tests with coverage report

# Code quality
black .                     # Formats code with Black formatter
flake8 .                   # Runs Flake8 linting
mypy .                     # Runs MyPy type checking
```

### Code Quality Standards
- **Frontend**: ESLint configuration with React best practices
- **Backend**: Black formatting, Flake8 linting, MyPy type checking
- **Testing**: Jest (frontend) and pytest (backend) with coverage requirements
- **Pre-commit hooks**: Automated code quality checks before commits

---

## 📡 API Documentation

### Core Endpoints
- **Patients**: `/api/patients` - Patient management and medical history
- **Surveys**: `/api/surveys` - Basic injury assessment surveys
- **Wearable Data**: `/api/wearable` - Device data integration and analytics
- **AI Insights**: `/api/insights` - ML-powered recovery analysis

### PRO Questionnaire APIs

#### KOOS (Knee Patients)
- `POST /api/koos` - Submit KOOS questionnaire (42 items, 0-4 scale)
- `GET /api/koos/{patient_id}` - Retrieve KOOS score history
- `GET /api/koos/{patient_id}/latest` - Get latest KOOS scores
- `GET /api/koos/{patient_id}/trends` - Analyze score trends over time

#### ASES (Shoulder Patients)  
- `POST /api/ases` - Submit ASES questionnaire (pain VAS + 10 function items)
- `GET /api/ases/{patient_id}` - Retrieve ASES score history  
- `GET /api/ases/{patient_id}/latest` - Get latest ASES scores
- `GET /api/ases/{patient_id}/trends` - Analyze score trends over time

### Authentication
- **JWT-based authentication** for secure API access
- **Role-based permissions** (patient, provider, researcher, admin)
- **Token refresh** for long-term sessions

**Interactive API Documentation**: Available at `http://localhost:8001/docs` when backend is running

---

## 🔧 Troubleshooting

### Common Development Issues

#### Backend Connection Problems
```bash
# Check MongoDB connection
mongo mongodb://localhost:27017/ortho_recovery

# Verify environment variables
cat backend/.env

# Check Python dependencies
pip list | grep fastapi
```

#### Frontend Build Issues
```bash
# Clear node modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Check for port conflicts
lsof -i :3000

# Verify environment variables
cat frontend/.env
```

#### Docker Deployment Issues
```bash
# Check container logs
docker logs <container-id>

# Verify image build
docker images | grep rcvryai

# Check port bindings
docker ps -a
```

### Environment Variable Requirements

#### Backend (.env)
```
MONGO_URL=mongodb://localhost:27017
DB_NAME=ortho_recovery
JWT_SECRET=your-secure-secret-key
API_BASE_URL=http://localhost:8001
```

#### Frontend (.env)
```
REACT_APP_BACKEND_URL=http://localhost:8001
```

### Performance Optimization
- **Frontend**: Use React DevTools Profiler for component optimization
- **Backend**: Monitor FastAPI logs for slow endpoints
- **Database**: Create indexes for frequently queried fields
- **Wearable Data**: Implement data pagination for large datasets

---

## 📚 Additional Resources

- **Clinical Validation**: KOOS and ASES scoring algorithms follow published clinical standards
- **Data Privacy**: HIPAA-compliant data handling and storage practices
- **Research Compliance**: Validated datasets suitable for clinical research
- **Integration**: REST APIs support EMR and wearable device integration

---

## 📞 Support & Contributing

For questions, issues, or contributions:
- **Issues**: GitHub Issues tracker
- **Documentation**: Inline code documentation and API specs
- **Clinical Questions**: Consult with orthopedic specialists for clinical validation

---

*RcvryAI OrthoTrack - Advancing orthopedic care through data-driven recovery monitoring*