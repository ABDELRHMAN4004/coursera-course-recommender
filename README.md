# 🎓 Coursera Course Recommendation System

An interactive **content-based recommendation system** that helps users discover Coursera courses based on their learning goals and preferences.

## 🌐 Live Demo

Coming soon.

## ✨ Features

* 🔎 Free-text learning goal
* 🎯 Subject filtering
* 📚 Level filtering
* 🏫 Institution filtering
* 🎓 Learning product filtering
* ⏱️ Duration filtering
* ⭐ Minimum rating
* 💬 Minimum reviews
* 🔢 Custom number of recommendations
* ⚖️ Custom ranking weights
* 🤖 TF-IDF-based content similarity
* 📊 Rating-based ranking
* 📈 Popularity-based ranking

---

## 🧠 Recommendation Approach

The system currently uses a **Content-Based Recommendation System**.

Each course is represented using information such as:

* Subject
* Course Title
* Institution
* Learning Product
* Level
* Duration
* Gained Skills

These features are combined into a textual **course profile**.

The course profiles are then transformed using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

**Cosine Similarity** is used to measure the similarity between the user's learning goal and the available courses.

The final recommendation ranking combines content similarity, course rating, and course popularity:

```text
Recommendation Score =
    Similarity Weight × Content Similarity
    +
    Rating Weight × Rating Score
    +
    Popularity Weight × Popularity Score
```

This allows users to control how much each component contributes to the final ranking.

---

## 📁 Project Structure

```text
coursera-course-recommender/
│
├── app/
│   └── app.py
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── recommender.py
│
├── data/
│   └── Coursera.csv
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* TF-IDF
* Cosine Similarity

---

## 📊 Dataset

The project uses the **Coursera Courses and Skills Dataset 2025**.

The dataset contains course information including:

* Subject
* Course Title
* Institution
* Learning Product
* Level
* Duration
* Gained Skills
* Rating
* Reviews

The dataset is used to build course profiles and generate personalized recommendations based on the user's input and selected preferences.

---

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/ABDELRHMAN4004/coursera-course-recommender.git
```

### 2. Enter the project directory

```bash
cd coursera-course-recommender
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run app/app.py
```

---

## 🔮 Future Improvements

* Machine-learning-based ranking
* Collaborative filtering
* Hybrid recommendation system
* User profiles
* Course history
* More personalized recommendations
* Recommendation evaluation metrics
* FastAPI backend
* Production deployment

---

## 👨‍💻 Author

**Abdelrhman Khalil Abdallah**

Data Science | Machine Learning | Deep Learning | Computer Vision

* GitHub: [ABDELRHMAN4004](https://github.com/ABDELRHMAN4004)
* Portfolio: [abdelrhman4004.github.io/portfolio.1](https://abdelrhman4004.github.io/portfolio.1/)
