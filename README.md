# Product Defect Detection API

This is a Flask-based API that uses a Random Forest model to predict whether a product is "Defective" or "Non-Defective" based on various input features such as Production Volume, Defect Rate, Quality Score, and Maintenance Hours.

## Features
- Upload a CSV file containing the product dataset.
- Train a machine learning model based on the uploaded dataset.
- Predict the defect status of a product based on input features.
- Confidence level in prediction is provided, with a threshold for determining defect status.

## Setup and Run

### Prerequisites
1. **Python 3.x** or higher
2. **pip** (Python package installer)

### Step-by-Step Setup

1. **Clone the repository**:
   
   git clone https://github.com/Thrylos13/Assignment.git
   cd Assignment
   

2. **Install dependencies**:
   Create and activate a virtual environment:

   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`


   Install the required libraries:
   
   pip install -r requirements.txt
  

3. **Run the Flask application**:
   After setting up the environment and installing dependencies, run the Flask application:
 
   python app.py
 
   This will start the server at `http://127.0.0.1:5000`.

## API Endpoints

### 1. Upload Dataset
**Endpoint**: `/upload`  
**Method**: `POST`  
**Description**: Upload a CSV file containing the dataset. The CSV file should include columns `ProductionVolume`, `DefectRate`, `QualityScore`, `MaintenanceHours`, and `DefectStatus`.

#### Step-by-Step in Postman:
1. Open **Postman**.
2. Set the **request type** to **POST**.
3. Enter the URL: `http://127.0.0.1:5000/upload`
4. Go to the **Body** tab, select **form-data**, and add the following key-value pair:
   - **Key**: `file`
   - **Value**: Choose a CSV file from your system (make sure the file has the appropriate columns).
5. Click **Send**.

**Response Example**:
```json
{
  "message": "File uploaded successfully",
  "columns": ["ProductionVolume", "DefectRate", "QualityScore", "MaintenanceHours", "DefectStatus"]
}
```

### 2. Train the Model
**Endpoint**: `/train`  
**Method**: `POST`  
**Description**: Train the machine learning model using the uploaded dataset. The model will be saved and can be used for predictions.

#### Step-by-Step in Postman:
1. Set the **request type** to **POST**.
2. Enter the URL: `http://127.0.0.1:5000/train`
3. Click **Send**.

**Response Example**:
```json
{
  "accuracy": 0.85,
  "f1_score": 0.83,
  "test_accuracy": 0.84
}
```

### 3. Predict Defect Status
**Endpoint**: `/predict`  
**Method**: `POST`  
**Description**: Predict the defect status of a product based on input features. The features include `ProductionVolume`, `DefectRate`, `QualityScore`, and `MaintenanceHours`. The prediction will return the defect status ("Defective" or "Non-Defective") along with the confidence level of the prediction.

#### Step-by-Step in Postman:
1. Set the **request type** to **POST**.
2. Enter the URL: `http://127.0.0.1:5000/predict`
3. Go to the **Body** tab, select **raw**, and set the type to **JSON**.
4. Enter the following JSON data with appropriate values:
   ```json
   {
     "ProductionVolume": 2000,
     "DefectRate": 0.03,
     "QualityScore": 88,
     "MaintenanceHours": 50
   }
   ```
5. Click **Send**.

**Response Example**:
```json
{
  "DefectStatus": "Defective",
  "Confidence": 89.56
}
```

### Logic for Defect Status
- **Defective**: The product is predicted to be defective if the confidence level is above 80%.
- **Non-Defective**: The product is predicted to be non-defective if the confidence level is below 80%.

### Notes
- The model is trained using a Random Forest classifier, and it evaluates the dataset based on four key features: `ProductionVolume`, `DefectRate`, `QualityScore`, and `MaintenanceHours`.
- The model is saved after training and loaded whenever predictions are made.

## Model Training Flow

1. **Data Upload**: Upload a CSV file that contains the product features and the defect status.
2. **Training**: Once the data is uploaded, call the `/train` endpoint to train the model. This will save the model as `model.pkl`.
3. **Prediction**: Use the `/predict` endpoint to get predictions for new data, with a confidence score.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
