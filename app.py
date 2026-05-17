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
# Streamlit UI
# -----------------------------

st.title("NFC Offline AI Assistant")

question = st.text_input(
    "Ask a question:"
)

# -----------------------------
# Question Answering
# -----------------------------

if question:

    # Convert question into vector
    question_vector = vectorizer.transform(
        [question]
    )

    # Similarity calculation
    similarities = cosine_similarity(
        question_vector,
        vectors
    )

    # Top matching chunks
    top_indices = similarities[0].argsort()[-3:][::-1]

    top_scores = similarities[0][top_indices]

    best_score = top_scores[0]

    # -----------------------------
    # Low confidence handling
    # -----------------------------

    if best_score < 0.15:

        st.warning(
            """
            I could not confidently find the answer
            in the available documents.

            Please contact support:

            support@nfc.com
            1800-000-123
            """
        )

    else:

        answer_sentences = []

        used_sources = set()

        # Important words from question
        question_words = [
            word.lower()
            for word in question.split()
            if len(word) > 3
        ]

        # -----------------------------
        # Process chunks
        # -----------------------------

        for i, index in enumerate(top_indices):

            if top_scores[i] > 0.15:

                chunk = chunks[index]

                source = sources[index]

                used_sources.add(source)

                # Split chunk into sentences
                sentences = chunk.split(".")

                for sentence in sentences:

                    sentence = sentence.strip()

                    # Ignore short sentences
                    if len(sentence) < 15:
                        continue

                    # Ignore headings/noise
                    if (
                        "guide" in sentence.lower()
                        or "training" in sentence.lower()
                    ):
                        continue

                    # Match count
                    match_count = sum(
                        1
                        for word in question_words
                        if word in sentence.lower()
                    )

                    # Keep relevant sentences only
                    if match_count >= 1:

                        answer_sentences.append(
                            sentence + "."
                        )

        # Remove duplicates
        final_answer = list(
            dict.fromkeys(answer_sentences)
        )

        # -----------------------------
        # Final display
        # -----------------------------

        if final_answer:

            st.subheader("Answer")

            for sentence in final_answer:

                st.write("- " + sentence)

            st.subheader("Sources")

            for source in used_sources:

                st.write("- " + source)

        else:

            st.warning(
                """
                I could not confidently find the answer
                in the available documents.

                Please contact support:

                support@nfc.com
                1800-000-123
                """
            )