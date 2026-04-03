# 🌾 Agricultural Crop Yield Prediction

An end-to-end Machine Learning project using Python and Streamlit to predict crop yield based on environmental and agricultural factors.

## 📁 Folder Structure

```text
project/
│
├── data/                       # Contains the dataset
│   └── crop_yield.csv          # The agricultural crop yield dataset
│
├── model/                      # ML Model code and saved artifacts
│   ├── train.py                # Script to preprocess data and train the model
│   ├── model.pkl               # Saved RandomForestRegressor model (generated)
│   ├── scaler.pkl              # Saved StandardScaler object (generated)
│   ├── metrics.pkl             # Saved metrics for accuracy and RMSE
│   └── encoders.pkl            # Saved LabelEncoders for categorical data (generated)
│
├── app.py                      # Streamlit frontend application
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## 🔄 Workflow

1. **Data Preprocessing**: The dataset is loaded, cleansed of missing values, and processed. Categorical features (`Area`, `Item`) are encoded, and numerical features (`Year`, `Rainfall`, `Pesticides`, `Temperature`) are scaled using a standard scaler.
2. **Model Training**: A `RandomForestRegressor` algorithm is trained on the processed data using Scikit-Learn.
3. **Artifact Preservation**: The trained machine learning model, data scaler, and categorical encoders are serialized and exported into `.pkl` format for later active inference.
4. **Interactive UI**: The Streamlit application (`app.py`) dynamically reads the `.pkl` files during startup, structures a user interface, and accurately transforms end-user input to perform live yield predictions in `hg/ha`.

## 🚀 How to Run Locally

### 1. Install Dependencies
Make sure you have Python installed. Install the required libraries via `pip`:
```bash
pip install -r requirements.txt
```

### 2. Insert the Dataset
Download the original dataset from Kaggle and place `crop_yield.csv` inside the designated `data/` directory. *(Note: If the document is omitted, running the training script will automatically assemble a synthetic mock dataset to verify your workflow works properly).*

### 3. Train the Model
Train the machine learning model from the project's root directory. This will output the newly generated `.pkl` files into your network!
```bash
python model/train.py
```

### 4. Run the Streamlit App
Start the interactive UI which automatically builds the interface onto a web browser!
```bash
streamlit run app.py
```

## ☁️ How to Deploy (Streamlit Community Cloud)

This project has been structurally optimized for continuous free deployment onto Streamlit's official cloud solution.

### Step 1: Upload Model to Google Drive
Because the model file (`model.pkl`) exceeds GitHub's 100 MB file size limit, it must be hosted externally:
1. Upload your generated `model/model.pkl` file to your Google Drive.
2. Right-click the file, select **Share**, and change General Access to **"Anyone with the link"**.
3. Copy the unique **File ID** from the generated link. 
*(For example, if the link is `https://drive.google.com/file/d/1o63pfzG-3S8OBkcKerJdpDwsqo8eT8rP/view`, the ID is `1o63pfzG-3S8OBkcKerJdpDwsqo8eT8rP`)*.
4. Open `app.py` in your code editor and replace the placeholder `YOUR_GOOGLE_DRIVE_FILE_ID` with your actual copied File ID.

### Step 2: Upload Code to GitHub
Initialize your Git repository and upload the remaining files to a remote public GitHub repository you own. 

The `.gitignore` has been pre-configured to ignore the massive `model.pkl` file so it won't crash your GitHub upload, but it **will** upload your other lightweight model files (`scaler.pkl`, `encoders.pkl`,`metrices.pkl`) and the `app.py` script.

```bash
# If you previously tried committing the large file, you MUST remove it from git's history before pushing:
git rm --cached model/model.pkl
git commit -m "Removed large model from tracking"

# Standard upload workflow
git init
git add .
git commit -m "Initial commit - Crop Yield Prediction"
git branch -M main
git remote add origin https://github.com/Crop-Yield-Estimation.git
git push -u origin main
```

### Step 2: Deploy to Streamlit Community Cloud
1. Configure a free account at **[Streamlit Community Cloud](https://share.streamlit.io/)** and authorize it to link securely with your GitHub account.
2. Click the **"New app"** button.
3. Select your designated public repository that hosts this project.
4. Set the **Branch** to `main`.
5. Specify the **Main file path** exactly as `app.py`.
6. Click **Deploy!**

Streamlit servers will systematically install all instances listed in `requirements.txt` and launch your project with a live globally accessible URL link!
URL : https://crop-yield-estimation-u46pahzx5zpjqnjmjgzohs.streamlit.app
Accuracy : 98.57
RMSE : 10179
