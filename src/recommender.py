import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CourseRecommender:
    def __init__(self, df):
        self.df = df.copy()

        # Build course profile
        self.df["course_profile"] = (
            self.df["Subject"].astype(str) + " " +
            self.df["Title"].astype(str) + " " +
            self.df["Institution"].astype(str) + " " +
            self.df["Learning Product"].astype(str) + " " +
            self.df["Level"].astype(str) + " " +
            self.df["Duration"].astype(str) + " " +
            self.df["Gained Skills"].astype(str)
        )

        # TF-IDF
        self.tfidf = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            ngram_range=(1, 2)
        )

        self.tfidf_matrix = self.tfidf.fit_transform(
            self.df["course_profile"]
        )

    def recommend(
        self,
        query="",
        subjects=None,
        levels=None,
        learning_products=None,
        institutions=None,
        durations=None,
        min_rating=0.0,
        min_reviews=0,
        n=10,
        similarity_weight=0.7,
        rating_weight=0.2,
        popularity_weight=0.1
    ):

        # Copy dataframe for filtering
        results = self.df.copy()

        # -------------------------
        # Filters
        # -------------------------

        if subjects:
            results = results[
                results["Subject"].isin(subjects)
            ]

        if levels:
            results = results[
                results["Level"].isin(levels)
            ]

        if learning_products:
            results = results[
                results["Learning Product"].isin(learning_products)
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

        if results.empty:
            return pd.DataFrame()

        # -------------------------
        # Content Similarity
        # -------------------------

        if query.strip():

            query_vector = self.tfidf.transform([query])

            similarities = cosine_similarity(
                query_vector,
                self.tfidf_matrix
            ).flatten()

            results["Similarity_Score"] = similarities[
                results.index
            ]

        else:
            results["Similarity_Score"] = 0.0

        # -------------------------
        # Rating Score
        # -------------------------

        results["Rating_Score"] = (
            results["Rate"] / 5
        )

        # -------------------------
        # Popularity Score
        # -------------------------

        max_reviews = results["Reviews_Log"].max()

        if max_reviews > 0:
            results["Popularity_Score"] = (
                results["Reviews_Log"] / max_reviews
            )
        else:
            results["Popularity_Score"] = 0.0

        # -------------------------
        # Final Recommendation Score
        # -------------------------

        results["Recommendation_Score"] = (
            similarity_weight * results["Similarity_Score"]
            + rating_weight * results["Rating_Score"]
            + popularity_weight * results["Popularity_Score"]
        )

        # Sort recommendations
        results = results.sort_values(
            by="Recommendation_Score",
            ascending=False
        )

        # Return top N
        return results.head(n)