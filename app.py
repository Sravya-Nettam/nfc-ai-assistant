import streamlit as st
import pickle

from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load vector store
# -----------------------------

with open("vector_store.pkl", "rb") as f:

    (
        vectorizer,
        vectors,
        chunks,
        sources
    ) = pickle.load(f)

# -----------------------------
# Streamlit Page Config
# -----------------------------

st.set_page_config(
    page_title="NFC Offline AI Assistant",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------

st.title("NFC Offline AI Assistant")

st.write(
    "Offline enterprise document assistant"
)

# -----------------------------
# User Question
# -----------------------------

question = st.text_input(
    "Ask a question:"
)

# -----------------------------
# Question Answering
# -----------------------------

if question:

    with st.spinner(
        "Searching documents..."
    ):

        # Convert question to vector
        question_vector = vectorizer.transform(
            [question]
        )

        # Similarity calculation
        similarities = cosine_similarity(
            question_vector,
            vectors
        )

        # Get top matches
        top_indices = similarities[0].argsort()[-3:][::-1]

        top_scores = similarities[0][top_indices]

        best_score = top_scores[0]

        # Low confidence
        if best_score < 0.05:

            st.warning(
                """
                I could not confidently find the answer
                in the available documents.

                Please contact support.
                """
            )

        else:

            answer_sentences = []

            question_words = [
                word.lower()
                for word in question.split()
                if len(word) > 3
            ]

            # Process relevant chunks
            for i, index in enumerate(top_indices):

                if top_scores[i] > 0.05:

                    chunk = chunks[index]

                    sentences = (
                        chunk
                        .replace("\n", ". ")
                        .split(".")
                    )

                    for sentence in sentences:

                        sentence = sentence.strip()

                        if len(sentence) < 20:
                            continue

                        lower_sentence = sentence.lower()

                        if (
                            "chapter" in lower_sentence
                            or "contents" in lower_sentence
                        ):
                            continue

                        match_count = sum(
                            1
                            for word in question_words
                            if word in lower_sentence
                        )

                        if match_count >= 1:

                            answer_sentences.append(
                                sentence + "."
                            )

            # Remove duplicates
            final_answer = list(
                dict.fromkeys(answer_sentences)
            )

            # Final output
            if final_answer:

                st.subheader("Answer")

                for sentence in final_answer[:5]:

                    st.write(
                        "- " + sentence
                    )

            else:

                st.warning(
                    """
                    Relevant information found,
                    but no exact answer sentence matched.
                    """
                )