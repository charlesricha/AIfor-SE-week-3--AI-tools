

<h1 align="center">🤖 AI for Sustainable Development – Practical Tasks</h1>

<p align="center">
This repository contains hands-on implementations of Classical Machine Learning, Deep Learning, and Natural Language Processing using real datasets and Python-based tools.
</p>

---

<!-- 🌐 Project Grid -->
<style>
  .task-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    justify-content: center;
    margin-top: 30px;
  }

  .task-box {
    flex: 1 1 300px;
    max-width: 300px;
    background: #f9f9f9;
    border: 1px solid #ddd;
    border-radius: 12px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.08);
    overflow: hidden;
    transition: transform 0.2s ease;
  }

  .task-box:hover {
    transform: scale(1.02);
    border-color: #7d55ff;
  }

  .task-box h2 {
    font-size: 1.3rem;
    text-align: center;
    background: #7d55ff;
    color: white;
    margin: 0;
    padding: 12px;
  }

  .task-box img {
    width: 100%;
    height: 200px;
    object-fit: cover;
    display: block;
  }

  .task-box p {
    padding: 12px;
    font-size: 0.95rem;
    color: #333;
    text-align: justify;
  }
</style>

<div class="task-grid">
  
  <!-- Task 1 -->
  <div class="task-box">
    <h2>Task 1: Classical ML – Iris Dataset</h2>
    <img src="./Media [images & videos]/Iris Dataset execution.PNG" alt="Task 1 - Iris Model">
    <p>
      Train a decision tree model on the Iris dataset using Scikit-learn. Includes preprocessing, label encoding, and evaluation with accuracy, precision, and recall.
    </p>
  </div>

  <!-- Task 2 -->
  <div class="task-box">
    <h2>Task 2: Deep Learning – MNIST & Streamlit App</h2>
    <img src="./Media [images & videos]/streamlit app.PNG" alt="Task 2 - Streamlit UI">
    <p>
      Build a CNN to classify handwritten digits with >95% accuracy. Includes model training, prediction on user images, and a live Streamlit web app with drag-and-drop UI.
    </p>
  </div>

  <!-- Task 3 -->
  <div class="task-box">
    <h2>Task 3: NLP with spaCy – Amazon Reviews</h2>
    <img src="./Media [images & videos]/task3-nlp-output.PNG" alt="Task 3 - NLP Output (replace if needed)">
    <p>
      Extract brand/product names using Named Entity Recognition (NER), and run rule-based sentiment analysis on user-provided Amazon product reviews using spaCy.
    </p>
  </div>

</div>



## 📚 Table of Contents

- [🧠 Task 1 – Classical ML with Scikit-learn](#-task-1--classical-ml-with-scikit-learn)
- [🧪 Task 2 – Deep Learning with TensorFlow](#-task-2--deep-learning-with-tensorflow)
- [🧰 Task 3 – Extension or Custom Feature](#-task-3--extension-or-custom-feature)
- [▶️ Running the Projects](#️-running-the-projects)
- [📁 Project Structure](#-project-structure)
- [📦 Requirements](#-requirements)

---

## 🧠 Task 1 – Classical ML with Scikit-learn

**Objective:**  
Use the Iris dataset to train a decision tree classifier that predicts the species of an iris flower.

**Features:**
- Load and preprocess data (cleaned automatically)
- Train a Decision Tree Classifier
- Evaluate model using accuracy, precision, and recall
- Display results in the terminal

**File:**  
`task1-sklearn.py`

---

## 🧪 Task 2 – Deep Learning with TensorFlow

**Objective:**  
Train a Convolutional Neural Network (CNN) on the MNIST dataset to classify handwritten digits with >95% accuracy.

**Features:**
- CNN model built using TensorFlow/Keras
- Trained on MNIST digits dataset
- Saved model as `mnist_cnn.h5`
- Predict on 5 test images
- Predict on custom external images (`predict_from_image.py`)
- Drag-and-drop web app using Streamlit (`app.py`)

**Files:**
- Training script: `task2-tensorflow.py`
- Image prediction: `predict_from_image.py`
- Web app: `app.py`

---

## 🧰 Task 3 – Extension or Custom Feature

*Optional: You can define your own task here (e.g., model deployment, API integration, or a custom dataset).*

---

## ▶️ Running the Projects

### 🔹 Task 1 (Scikit-learn)
```bash
python task1-sklearn.py
```

🔹 Task 2 (TensorFlow)

Train and save model:

```bash
python task2-tensorflow.py
```

Predict on external image:

```bash
python predict_from_image.py
```

Run the drag-and-drop web app:

```bash
streamlit run app.py
```

> Make sure to have an image named `your_digit.png` in the same folder before running prediction.

📁 Project Structure

```
/project-folder/
├── task1-sklearn.py
├── task2-tensorflow.py
├── mnist_cnn.h5              # saved after training
├── your_digit.png            # image to test predictions
├── predict_from_image.py
├── app.py                    # Streamlit UI
└── README.md
```

📦 Requirements

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

### `requirements.txt`
```
tensorflow
scikit-learn
numpy
matplotlib
pillow
streamlit
```

💬 License

Open source, educational use only.

