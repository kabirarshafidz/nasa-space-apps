# 🌌 NASA Exoplanet Detection & Visualization Platform

A full-stack machine learning application for detecting and classifying exoplanets using TESS (Transiting Exoplanet Survey Satellite) data. Features advanced ML models, interactive 3D visualizations, and real-time predictions.

![NASA Exoplanet Platform](https://img.shields.io/badge/NASA-Exoplanet%20Detection-blue)
![Python](https://img.shields.io/badge/Python-3.9+-green)
![Next.js](https://img.shields.io/badge/Next.js-15.5-black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal)

## 🚀 Features

### 🤖 Machine Learning

- **Multiple ML Models**: XGBoost, Random Forest, Logistic Regression
- **Cross-Validation Training**: Robust model evaluation with stratified K-fold
- **Advanced Preprocessing**: KNN imputation, standard scaling, PCA dimensionality reduction
- **Planet Type Classification**: Automated classification into Hot Jupiter, Rocky, Super-Earth, Gas Giant
- **Model Persistence**: Save and load trained models with metadata
- **Batch Predictions**: Process multiple candidates simultaneously

### 📊 Visualization & Analytics

- **Interactive 3D Solar System**: Real-time physics-based orbital simulations
- **Habitable Zone Visualization**: Visual indicators for potentially habitable planets
- **Performance Metrics**: ROC curves, confusion matrices, feature importance plots
- **Training History**: Track and compare model performance over time

### 💬 AI-Powered Insights

- **Exoplanet Chatbot**: Get instant insights about detected planets
- **Scientific Analysis**: AI-powered explanations of planetary characteristics
- **Real-time Predictions**: Interactive prediction interface with visual feedback

### 🔐 User Management

- **Authentication System**: Secure user login and registration
- **Session Management**: Persistent training sessions and model history
- **User-Specific Models**: Personal model storage and retrieval

## 🏗️ Architecture

```
nasa/
├── api/                    # FastAPI Backend
│   ├── main.py            # Main API endpoints
│   ├── train_pipeline.py  # ML training pipeline
│   ├── models/            # Trained model storage
│   ├── artifacts/         # PCA/KNN classification artifacts
│   └── content/           # TESS data files
│
├── fe/                    # Next.js Frontend
│   └── src/
│       ├── app/
│       │   ├── predict/   # Prediction interface
│       │   ├── train/     # Model training UI
│       │   └── 3d-visualization/  # 3D solar system
│       └── components/    # Reusable UI components
│
├── data/                  # Raw datasets
│   └── tess.csv          # TESS exoplanet data
│
└── notebooks/            # Jupyter notebooks for EDA
```

## 🛠️ Tech Stack

### Backend

- **FastAPI**: High-performance async API framework
- **Scikit-learn**: ML preprocessing and model training
- **XGBoost**: Gradient boosting for classification
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Matplotlib & Seaborn**: Visualization and chart generation
- **Joblib**: Model serialization
- **Boto3**: Optional S3/R2 cloud storage

### Frontend

- **Next.js 15**: React framework with App Router
- **React Three Fiber**: 3D graphics with Three.js
- **Tailwind CSS**: Utility-first styling
- **shadcn/ui**: Beautiful UI components
- **Recharts**: Data visualization
- **Better Auth**: Authentication system
- **AI SDK**: OpenAI integration for chatbot
- **Drizzle ORM**: Database management

## 📋 Prerequisites

- **Python**: 3.9 or higher
- **Node.js**: 18 or higher
- **npm** or **pnpm**: Latest version
- **PostgreSQL**: For user authentication (optional)

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd nasa
```

### 2. Backend Setup

#### Option A: Using Virtual Environment

```bash
cd api
python3 -m venv .
source bin/activate  # On Windows: bin\Activate.ps1
pip install -r requirements.txt
```

#### Option B: Using uv (recommended)

```bash
cd api
uv pip sync requirements.txt
```

### 3. Frontend Setup

```bash
cd fe
npm install
# or
pnpm install
```

### 4. Environment Variables

Create a `.env` file in the `fe/` directory:

```env
# Database (optional - for authentication)
DATABASE_URL=postgresql://user:password@localhost:5432/nasa

# OpenAI API (for chatbot)
OPENAI_API_KEY=your_openai_api_key

# Better Auth
BETTER_AUTH_SECRET=your_secret_key
BETTER_AUTH_URL=http://localhost:3000

# API URL
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Create a `.env` file in the `api/` directory (optional - for cloud storage):

```env
# AWS S3 or Cloudflare R2 (optional)
R2_ENDPOINT_URL=your_r2_endpoint
R2_ACCESS_KEY_ID=your_access_key
R2_SECRET_ACCESS_KEY=your_secret_key
R2_BUCKET_NAME=your_bucket_name
```

## 🚀 Running the Application

### Start the Backend API

```bash
cd api
uvicorn main:app --reload --port 8000
```

Or use the provided script:

```bash
cd api
./start_api.sh
```

### Start the Frontend

```bash
cd fe
npm run dev
# or
pnpm dev
```

The application will be available at:

- **Frontend**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📖 Usage

### 1. Train a Model

1. Navigate to the **Train** page
2. Upload your TESS CSV file (or use the default dataset)
3. Configure model parameters:
   - Choose model type (XGBoost, Random Forest, Logistic Regression)
   - Set hyperparameters
   - Configure cross-validation settings
4. Start training and monitor progress
5. Review metrics, charts, and feature importance
6. Save the model for future predictions

### 2. Make Predictions

1. Navigate to the **Predict** page
2. Select a trained model from your history
3. Upload a CSV file with candidate exoplanet data
4. View predictions with confidence scores
5. Explore 3D visualizations of predicted planets
6. Chat with the AI to learn more about the results

### 3. Visualize Exoplanets

1. Go to the **3D Visualization** page
2. Interactive orbital mechanics simulation
3. Zoom, rotate, and explore planetary systems
4. View habitable zones and planetary characteristics
5. Real-time physics calculations

## 🔬 API Endpoints

### Training Endpoints

- `POST /train/upload-csv` - Upload training data
- `POST /train/start-training` - Start model training
- `GET /train/training-status/{task_id}` - Check training progress
- `GET /training/entries` - List all training sessions
- `POST /training/save-result` - Save training results

### Prediction Endpoints

- `POST /predict` - Make predictions on new data
- `POST /predict/batch-csv` - Batch predictions from CSV
- `GET /models` - List available models
- `GET /models/{model_name}` - Get model details

### Classification Endpoints

- `POST /classify-planet-types` - Classify planets into types
- `GET /tess-data` - Fetch TESS dataset
- `POST /chart/{chart_type}` - Generate visualization charts

### Model Management

- `GET /models/list` - List all saved models
- `DELETE /models/{model_name}` - Delete a model
- `GET /download-model/{model_name}` - Download model file

## 📊 Dataset

The project uses TESS (Transiting Exoplanet Survey Satellite) data, which includes:

- Planetary characteristics (radius, mass, temperature)
- Orbital parameters (period, semi-major axis)
- Stellar properties (temperature, radius, luminosity)
- Discovery metadata and flags

Key features used for classification:

- `pl_rade`: Planet radius (Earth radii)
- `pl_insol`: Insolation flux (Earth flux)
- `pl_eqt`: Equilibrium temperature (K)
- `pl_orbper`: Orbital period (days)
- `st_teff`: Stellar effective temperature (K)
- `st_rad`: Stellar radius (Solar radii)

## 🧪 Model Performance

The platform supports three classification algorithms:

| Model               | Typical Accuracy | Training Time | Best For             |
| ------------------- | ---------------- | ------------- | -------------------- |
| XGBoost             | 90-95%           | Medium        | Balanced performance |
| Random Forest       | 88-92%           | Fast          | Quick prototyping    |
| Logistic Regression | 85-88%           | Very Fast     | Baseline comparison  |

_Performance varies based on dataset size and hyperparameters_

## 🎨 UI Features

- **Dark Mode**: NASA-inspired dark theme with cosmic aesthetics
- **Responsive Design**: Mobile-first approach with adaptive layouts
- **Real-time Updates**: Live training progress and metrics
- **Interactive Charts**: Hover, zoom, and explore visualizations
- **Drag & Drop**: Easy file uploads
- **Accessible**: WCAG 2.1 AA compliant components

## 🐛 Known Issues & Limitations

- Large datasets (>100k rows) may require increased memory
- 3D visualization performance depends on GPU capabilities
- Chatbot requires OpenAI API key
- Authentication requires PostgreSQL database

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Development Notes

### Running Tests

```bash
# Backend tests
cd api
pytest

# Frontend tests
cd fe
npm test
```

### Code Quality

```bash
# Python linting
cd api
flake8 .
black .

# TypeScript linting
cd fe
npm run lint
```

### Database Migrations

```bash
cd fe
npx drizzle-kit push
```

## 🔮 Future Enhancements

- [ ] Support for additional datasets (Kepler, K2)
- [ ] Advanced feature engineering pipeline
- [ ] Model ensemble predictions
- [ ] Real-time data streaming
- [ ] Mobile app version
- [ ] Multi-language support
- [ ] Collaborative training sessions
- [ ] Custom model architecture builder

## 📚 Resources

- [TESS Mission](https://tess.mit.edu/)
- [NASA Exoplanet Archive](https://exoplanetarchive.ipasa.nasa.gov/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Next.js Documentation](https://nextjs.org/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- NASA for TESS mission data
- Scikit-learn community for ML tools
- Next.js and Vercel teams for frontend framework
- Three.js for 3D visualization capabilities

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub.

---

**Made with 🚀 for space enthusiasts and data scientists**
