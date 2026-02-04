from __future__ import annotations

from pathlib import Path
import zipfile

import pandas as pd
import streamlit as st

DATA_ZIP = Path(__file__).resolve().parent / "archive (2).zip"
TRAINING_FILE = "Training.csv"
DESCRIPTION_FILE = "disease_description.csv"
PRECAUTION_FILE = "disease_precaution.csv"


@st.cache_data
def load_data() -> tuple[pd.Index, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not DATA_ZIP.exists():
        raise FileNotFoundError(
            f"Expected dataset zip at {DATA_ZIP}. Please add the Kaggle dataset zip."
        )

    with zipfile.ZipFile(DATA_ZIP) as archive:
        training = pd.read_csv(archive.open(TRAINING_FILE))
        descriptions = pd.read_csv(archive.open(DESCRIPTION_FILE))
        precautions = pd.read_csv(archive.open(PRECAUTION_FILE))

    symptom_columns = training.columns.drop("prognosis")
    disease_symptoms = training.groupby("prognosis")[symptom_columns].max()
    return symptom_columns, disease_symptoms, descriptions, precautions


def build_symptom_options(symptom_columns: pd.Index) -> dict[str, str]:
    return {column.replace("_", " ").title(): column for column in symptom_columns}


def score_diseases(
    disease_symptoms: pd.DataFrame, selected_symptoms: list[str]
) -> pd.Series:
    if not selected_symptoms:
        return pd.Series(dtype=float)

    scores = disease_symptoms[selected_symptoms].sum(axis=1) / len(selected_symptoms)
    return scores.sort_values(ascending=False)


def get_description(descriptions: pd.DataFrame, disease: str) -> str | None:
    match = descriptions.loc[descriptions["Disease"] == disease]
    if match.empty:
        return None
    return match.iloc[0]["Symptom_Description"]


def get_precautions(precautions: pd.DataFrame, disease: str) -> list[str]:
    match = precautions.loc[precautions["Disease"] == disease]
    if match.empty:
        return []
    row = match.iloc[0]
    precaution_columns = [col for col in precautions.columns if col != "Disease"]
    return [
        value
        for value in (row[col] for col in precaution_columns)
        if isinstance(value, str) and value.strip()
    ]


st.set_page_config(page_title="Medical Symptom Checker", page_icon="🩺", layout="wide")

st.title("🩺 Medical Symptom Checker")
st.caption(
    "Educational demo: predictions are based on symptom overlap from a Kaggle dataset."
)

try:
    symptom_columns, disease_symptoms, descriptions, precautions = load_data()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

symptom_options = build_symptom_options(symptom_columns)
selected_labels = st.multiselect(
    "Select symptoms that apply", sorted(symptom_options.keys())
)
selected_symptoms = [symptom_options[label] for label in selected_labels]

st.divider()

if not selected_symptoms:
    st.info("Choose at least one symptom to see suggested conditions.")
else:
    scores = score_diseases(disease_symptoms, selected_symptoms)
    matches = scores[scores > 0].head(5)

    if matches.empty:
        st.warning("No matching conditions found. Try selecting different symptoms.")
    else:
        st.subheader("Top matches")
        for disease, score in matches.items():
            matched_count = int(score * len(selected_symptoms))
            st.markdown(
                f"**{disease}** — {matched_count}/{len(selected_symptoms)} symptoms matched"
            )
            description = get_description(descriptions, disease)
            if description:
                st.write(description)

            precaution_list = get_precautions(precautions, disease)
            if precaution_list:
                st.write("**Suggested precautions:**")
                st.markdown("\n".join(f"- {item}" for item in precaution_list))
            st.divider()

with st.sidebar:
    st.header("Dataset details")
    st.write(f"Symptoms available: {len(symptom_columns)}")
    st.write(f"Conditions covered: {len(disease_symptoms)}")
    st.write("Data source: Kaggle Disease Symptom Prediction dataset.")
    st.warning(
        "This tool does not provide medical advice. Always consult a licensed provider."
    )
