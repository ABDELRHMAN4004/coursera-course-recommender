
import streamlit as st
import pandas as pd
import numpy as np

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing import load_and_preprocess_data
from src.recommender import CourseRecommender

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Personalized Course Recommendation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 25px;
    }

    .course-title {
        font-size: 22px;
        font-weight: 650;
    }

    .match-score {
        font-size: 30px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD DATA
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DATA_PATH = ROOT_DIR / "data" / "Coursera.csv"

@st.cache_data
def load_data():
    return load_and_preprocess_data(DATA_PATH)

df = load_data()


# =========================================================
# BUILD TF-IDF MODEL
# =========================================================

@st.cache_resource
def build_tfidf_model(df):

    course_profile = (
        df["Subject"] + " " +
        df["Title"] + " " +
        df["Institution"] + " " +
        df["Learning Product"] + " " +
        df["Level"] + " " +
        df["Duration"] + " " +
        df["Gained Skills"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(
        course_profile
    )

    return vectorizer, tfidf_matrix


vectorizer, tfidf_matrix = build_tfidf_model(df)


# =========================================================
# RECOMMENDATION ENGINE
# =========================================================

def recommend_courses(
    query,
    subjects,
    levels,
    products,
    institutions,
    durations,
    min_rating,
    min_reviews,
    similarity_weight,
    rating_weight,
    popularity_weight,
    number_of_recommendations
):

    # -----------------------------------------------------
    # Query → TF-IDF
    # -----------------------------------------------------

    query_vector = vectorizer.transform(
        [query]
    )

    similarity_scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()


    # -----------------------------------------------------
    # Copy dataset
    # -----------------------------------------------------

    results = df.copy()

    results["Similarity"] = similarity_scores


    # -----------------------------------------------------
    # Apply filters
    # -----------------------------------------------------

    if subjects:

        results = results[
            results["Subject"].isin(subjects)
        ]


    if levels:

        results = results[
            results["Level"].isin(levels)
        ]


    if products:

        results = results[
            results["Learning Product"].isin(products)
        ]


    if institutions:

        results = results[
            results["Institution"].isin(institutions)
        ]


    if durations:

        results = results[
            results["Duration"].isin(durations)
        ]


    results = results[
        results["Rate"] >= min_rating
    ]


    results = results[
        results["Reviews"] >= min_reviews
    ]


    # -----------------------------------------------------
    # Check results
    # -----------------------------------------------------

    if results.empty:

        return results


    # -----------------------------------------------------
    # Rating score
    # -----------------------------------------------------

    results["Rating_Score"] = (
        results["Rate"] / 5.0
    )


    # -----------------------------------------------------
    # Popularity score
    # -----------------------------------------------------

    results["Popularity_Score"] = np.log1p(
        results["Reviews"]
    )

    max_popularity = results[
        "Popularity_Score"
    ].max()

    if max_popularity > 0:

        results["Popularity_Score"] = (
            results["Popularity_Score"]
            / max_popularity
        )

    else:

        results["Popularity_Score"] = 0


    # -----------------------------------------------------
    # Final recommendation score
    # -----------------------------------------------------

    total_weight = (
        similarity_weight
        + rating_weight
        + popularity_weight
    )

    if total_weight == 0:

        total_weight = 1


    results["Recommendation_Score"] = (

        (
            results["Similarity"]
            * similarity_weight
        )

        +

        (
            results["Rating_Score"]
            * rating_weight
        )

        +

        (
            results["Popularity_Score"]
            * popularity_weight
        )

    ) / total_weight


    # -----------------------------------------------------
    # Sort
    # -----------------------------------------------------

    results = results.sort_values(
        "Recommendation_Score",
        ascending=False
    )


    return results.head(
        number_of_recommendations
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '🎓 Personalized Course Recommendation System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Tell us what you want to learn and customize your
    preferences to discover the most suitable courses.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎯 Your Preferences")

st.sidebar.caption(
    "Customize your learning preferences."
)


# =========================================================
# SUBJECT
# =========================================================

subjects = sorted(
    [
        value
        for value in df["Subject"].unique()
        if value
    ]
)

selected_subjects = st.sidebar.multiselect(
    "📚 Subject",
    subjects,
    placeholder="Choose subjects..."
)


# =========================================================
# LEVEL
# =========================================================

levels = sorted(
    [
        value
        for value in df["Level"].unique()
        if value
    ]
)

selected_levels = st.sidebar.multiselect(
    "🎓 Level",
    levels,
    placeholder="Choose levels..."
)


# =========================================================
# LEARNING PRODUCT
# =========================================================

products = sorted(
    [
        value
        for value in df["Learning Product"].unique()
        if value
    ]
)

selected_products = st.sidebar.multiselect(
    "📦 Learning Product",
    products,
    placeholder="Course / Specialization..."
)


# =========================================================
# INSTITUTION
# =========================================================

institutions = sorted(
    [
        value
        for value in df["Institution"].unique()
        if value
    ]
)

selected_institutions = st.sidebar.multiselect(
    "🏫 Institution",
    institutions,
    placeholder="Choose institutions..."
)


# =========================================================
# DURATION
# =========================================================

durations = sorted(
    [
        value
        for value in df["Duration"].unique()
        if value
    ]
)

selected_durations = st.sidebar.multiselect(
    "⏱️ Duration",
    durations,
    placeholder="Choose durations..."
)


# =========================================================
# MINIMUM RATING
# =========================================================

min_rating = st.sidebar.slider(
    "⭐ Minimum Rating",
    min_value=0.0,
    max_value=5.0,
    value=0.0,
    step=0.1
)


# =========================================================
# MINIMUM REVIEWS
# =========================================================

max_reviews = int(
    df["Reviews"].max()
)

min_reviews = st.sidebar.number_input(
    "👥 Minimum Reviews",
    min_value=0,
    max_value=max_reviews,
    value=0,
    step=100
)


# =========================================================
# NUMBER OF RECOMMENDATIONS
# =========================================================

number_of_recommendations = st.sidebar.slider(
    "🔢 Number of Recommendations",
    min_value=3,
    max_value=20,
    value=10
)


# =========================================================
# MAIN QUERY
# =========================================================

st.subheader("🔍 What do you want to learn?")


query = st.text_area(
    "Describe your learning goal",
    placeholder=(
        "Example:\n"
        "I want to learn Machine Learning and "
        "Deep Learning using Python."
    ),
    height=120
)


# =========================================================
# QUICK TOPIC BUTTONS
# =========================================================

st.markdown("##### 💡 Quick Topics")

quick_topics = [
    "Python",
    "Data Science",
    "Machine Learning",
    "Deep Learning",
    "Computer Vision",
    "Artificial Intelligence",
    "Data Analysis"
]

topic_columns = st.columns(
    len(quick_topics)
)

for column, topic in zip(
    topic_columns,
    quick_topics
):

    with column:

        if st.button(
            topic,
            use_container_width=True
        ):

            st.session_state[
                "selected_topic"
            ] = topic


if "selected_topic" in st.session_state:

    if not query.strip():

        query = st.session_state[
            "selected_topic"
        ]


# =========================================================
# ADVANCED RANKING SETTINGS
# =========================================================

with st.expander(
    "⚙️ Advanced Ranking Settings"
):

    st.write(
        "Control how each factor affects the final ranking."
    )

    similarity_weight = st.slider(
        "🧠 Content Similarity",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.5
    )

    rating_weight = st.slider(
        "⭐ Rating",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.5
    )

    popularity_weight = st.slider(
        "👥 Popularity",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.5
    )


# =========================================================
# RECOMMEND BUTTON
# =========================================================

recommend_button = st.button(
    "🚀 Recommend Courses",
    type="primary",
    use_container_width=True
)


# =========================================================
# GENERATE RECOMMENDATIONS
# =========================================================

if recommend_button:

    if not query.strip():

        st.warning(
            "Please describe what you want to learn."
        )

    else:

        with st.spinner(
            "🔎 Finding the best courses for you..."
        ):

            recommendations = recommend_courses(

                query=query,

                subjects=selected_subjects,

                levels=selected_levels,

                products=selected_products,

                institutions=selected_institutions,

                durations=selected_durations,

                min_rating=min_rating,

                min_reviews=min_reviews,

                similarity_weight=
                    similarity_weight,

                rating_weight=
                    rating_weight,

                popularity_weight=
                    popularity_weight,

                number_of_recommendations=
                    number_of_recommendations
            )


        if recommendations.empty:

            st.error(
                """
                No courses matched your preferences.

                Try removing some filters or lowering
                the minimum rating/review requirements.
                """
            )

        else:

            st.session_state[
                "recommendations"
            ] = recommendations

            st.session_state[
                "search_query"
            ] = query


# =========================================================
# RESULTS
# =========================================================

if "recommendations" in st.session_state:

    recommendations = st.session_state[
        "recommendations"
    ]

    st.divider()

    st.subheader(
        "🎯 Recommended Courses For You"
    )

    st.caption(
        "Based on your learning goal and preferences:"
    )

    st.info(
        st.session_state[
            "search_query"
        ]
    )


    # =====================================================
    # SUMMARY
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Courses",
            len(recommendations)
        )

    with col2:

        st.metric(
            "Avg. Rating",
            f"{recommendations['Rate'].mean():.2f}"
        )

    with col3:

        st.metric(
            "Avg. Reviews",
            f"{int(recommendations['Reviews'].mean()):,}"
        )

    with col4:

        st.metric(
            "Top Match",
            f"{recommendations['Recommendation_Score'].iloc[0]:.0%}"
        )


    st.markdown("")


    # =====================================================
    # COURSE CARDS
    # =====================================================

    for rank, (_, row) in enumerate(
        recommendations.iterrows(),
        start=1
    ):

        with st.container(
            border=True
        ):

            left, right = st.columns(
                [5, 1]
            )


            # -------------------------------------------------
            # COURSE INFORMATION
            # -------------------------------------------------

            with left:

                st.markdown(
                    f'<div class="course-title">'
                    f'{rank}. {row["Title"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.write(
                    f"🏫 **Institution:** "
                    f"{row['Institution']}"
                )

                st.write(
                    f"📚 **Subject:** "
                    f"{row['Subject']}"
                )

                st.write(
                    f"🎓 **Level:** {row['Level']}   |   "
                    f"📦 **Type:** {row['Learning Product']}"
                )

                st.write(
                    f"⏱️ **Duration:** {row['Duration']}"
                )

                st.write(
                    f"⭐ **Rating:** {row['Rate']}   |   "
                    f"👥 **Reviews:** "
                    f"{int(row['Reviews']):,}"
                )

                if row["Gained Skills"]:

                    st.write(
                        f"🧠 **Skills:** "
                        f"{row['Gained Skills']}"
                    )


            # -------------------------------------------------
            # SCORE
            # -------------------------------------------------

            with right:

                st.markdown(
                    '<div class="match-score">'
                    f'{row["Recommendation_Score"]:.0%}'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.caption(
                    "Match Score"
                )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Course Recommendation System "
    "• TF-IDF • Cosine Similarity "
    "• Personalized Ranking"
)
