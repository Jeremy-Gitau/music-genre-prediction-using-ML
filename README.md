# 🎵 Music Genre Prediction using Machine Learning

This project demonstrates how to use different **machine learning algorithms** to predict a person’s preferred **music genre** based on their **age** and **gender**.

It explores multiple ML models such as:
- **Decision Tree Classifier**
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**
- **Random Forest Classifier**

Finally, it provides a simple **Gradio UI** for making predictions interactively.

---

## 📂 Project Structure

```

music-genre-prediction-using-ML/
│── music/
│   ├── music.csv                # Dataset (age, gender, genre)
│   ├── music\_prediction.ipynb   # Main notebook
│── README.md

````

---

## 📊 Dataset

The dataset (`music.csv`) contains 3 columns:

- `age` → Age of the listener  
- `gender` → 0 = Female, 1 = Male  
- `genre` → Preferred music genre (e.g., HipHop, Dance, Acoustic, etc.)

Example:

| age | gender | genre  |
|-----|--------|--------|
| 20  | 1      | HipHop |
| 23  | 1      | HipHop |
| 25  | 1      | Jazz   |

---

## 🚀 Installation & Setup

1. Clone this repository:
```bash
git clone https://github.com/Jeremy-Gitau/music-genre-prediction-using-ML.git
cd music-genre-prediction-using-ML/music
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, install manually:

```bash
pip install pandas scikit-learn matplotlib gradio
```

3. Run the Jupyter notebook:

```bash
jupyter notebook music_prediction.ipynb
```

---

## ⚙️ Usage

### Training Models

The notebook trains multiple models and evaluates them with **accuracy scores**:

* **Decision Tree Classifier** → 100% (overfitting possible)
* **KNN** → \~50%
* **Logistic Regression** → \~83%
* **Random Forest** → \~83%

### Making Predictions

Example:

```python
model.predict([[21,1],[22,0]])
```

Output:

```
['HipHop', 'Dance']
```

---

## 🎛 Gradio Interface

The notebook includes a **Gradio demo** to interact with the model:

```python
import gradio as gr

def pred(age, gender):
    prediction = model.predict([[age, gender]])
    return prediction

iface = gr.Interface(
    fn=pred, 
    inputs=[gr.Slider(0,70), gr.Checkbox(label="Male (1) / Female (0)")],
    outputs="text"
)
iface.launch()
```

This will launch a local web app where you can move the **age slider** and check/uncheck **gender** to predict the music genre.

⚠️ Note: If you see errors with `anyio` or `MissingSchema`, update your Gradio and Uvicorn installation:

```bash
pip install --upgrade gradio uvicorn anyio
```

---

## 📈 Results

| Model               | Accuracy |
| ------------------- | -------- |
| Decision Tree       | 1.00     |
| KNN                 | 0.50     |
| Logistic Regression | 0.83     |
| Random Forest       | 0.83     |

---

## 🔮 Future Improvements

* Use a **larger dataset** for better generalization.
* Apply **data preprocessing** (scaling, encoding, etc.) for models like Logistic Regression & KNN.
* Deploy the model as an API (FastAPI/Flask).
* Improve Gradio interface with better input controls.

---

## 👨‍💻 Author

**Jeremy Gitau**

* 💼 Software Engineer (AI, ML, Backend, Flutter)
* 🌍 Based in Kenya
* 🔗 [GitHub](https://github.com/Jeremy-Gitau)


---
