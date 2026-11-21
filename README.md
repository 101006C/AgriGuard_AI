# AgriGuard AI – Crop Disease Screening Prototype

CSE471 course project starter implementation.

## What is included
- React + Vite frontend
- Flask API
- MobileNetV2 transfer-learning training script
- Image upload and prediction endpoint
- Prediction history saved as JSONL
- Basic cautious next-step guidance
- Evaluation script

## Important limitations
1. This repository does not include trained model weights. Train the model before using prediction.
2. The initial model is a **crop disease image classifier**, not a pest detector. Pest detection requires a separate pest dataset and detection/classification model.
3. A leaf-image classifier cannot reliably determine field-level disease severity. Do not describe the severity feature as implemented.
4. The app is a screening aid, not an agricultural diagnosis. Confirm results with an agricultural expert.
5. Do not report accuracy until you have evaluated the model on an independent test set.

## 1. Requirements
- Python 3.10 or 3.11 recommended for TensorFlow compatibility
- Node.js 20 or newer
- VS Code
- Internet connection for installing packages and downloading dataset

## 2. Dataset setup
Use a dataset whose license permits your academic use, such as PlantVillage. Download and extract it. Organize images into class-named folders:

```text
AgriGuard_AI/
  dataset/
    Tomato___Early_blight/
      image1.jpg
    Tomato___healthy/
      image2.jpg
    Potato___Late_blight/
      image3.jpg
```

Each class folder should contain images belonging to that class. Do not mix train/test images from the same original image or near-duplicate images. For rigorous evaluation, create an independent test set before training. The included training script reserves a validation subset; it does not automatically create an independent test set.

## 3. Install Python dependencies
From the project root:

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install:
```bash
pip install -r backend/requirements.txt
```

## 4. Train the model
From the project root:

```bash
python model/train_model.py --data dataset --epochs 12 --output model
```

This creates:
- `model/crop_disease_model.keras`
- `model/class_names.json`

Training may take time and benefits from a GPU. If your laptop does not have a suitable GPU, Google Colab can be used. Download the two generated artifacts and place them in the local `model/` folder.

## 5. Start backend
Open a terminal at the project root:

```bash
python backend/app.py
```

Backend URL: `http://localhost:5000`

Check:
- `http://localhost:5000/api/health`

## 6. Start frontend
Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL, normally `http://localhost:5173`.

If the backend is hosted somewhere other than localhost, create `frontend/.env`:
```env
VITE_API_URL=https://your-backend-domain
```
Restart Vite after changing environment variables.

## 7. Evaluate the model
Prepare a separate test folder with the same class-subfolder structure, then run:

```bash
python model/evaluate_model.py --data dataset_test --model model/crop_disease_model.keras --classes model/class_names.json
```

Save the actual classification report and confusion matrix for the report. Do not invent results.

## 8. API endpoints
- `GET /api/health` – backend/model status
- `POST /api/predict` – multipart form upload with field name `image`
- `GET /api/history` – recent prediction history

## 9. Sample data requirement
The course guideline asks for 20–50 representative sample records or files and a data dictionary. Add 20–50 representative, properly licensed sample images and/or actual prediction records to `data/` before submission. The starter does not fabricate prediction records.

## 10. Deployment
Frontend can be deployed to Vercel or Netlify. Backend can be deployed to a Python-capable service such as Render, subject to its current free-tier and storage limitations. A TensorFlow model can be large; verify the hosting service's memory, build-time, and file-size limits. For a course demo, Hugging Face Spaces may be another option. Configure the frontend `VITE_API_URL` to the deployed backend URL and configure CORS for the deployed frontend domain before public release.

Do not deploy secrets or credentials. Keep model files and uploaded user images under appropriate access controls. Test the public URL before submission and keep it active through evaluation.

## 11. Project scope and future work
Current prototype: image-based crop disease classification for classes present in the training dataset.
Future work: dedicated pest detection model, severity segmentation, Grad-CAM explanations, multilingual interface, offline mobile support, weather/region-aware advisory with verified agricultural sources.

## 12. Academic integrity
Document dataset source, license, preprocessing, model settings, test split, and actual results. Cite dataset and research sources in the required format. Explain which parts were implemented and which remain future work.
