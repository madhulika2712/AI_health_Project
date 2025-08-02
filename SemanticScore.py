#pip install sentence-transformers pandas
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 🧠 Wrapper function to compute similarity
def compute_semantic_similarity(text1, text2):
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2))

# 📄 Function to compare batch from CSV and save results
def compare_answers_from_csv(input_csv_path, output_csv_path,
                             col_ground_truth='GroundTruth', col_generated='GeneratedAnswer'):
    """
    input_csv_path: CSV with ground truth and generated answers.
    output_csv_path: Where to save results with similarity score.
    col_ground_truth: Column name for reference text.
    col_generated: Column name for model/generated answers.
    """
    # Load data
    df = pd.read_csv(input_csv_path)

    # Compute semantic similarity
    scores = []
    for idx, row in df.iterrows():
        gt = str(row[col_ground_truth])
        gen = str(row[col_generated])
        score = compute_semantic_similarity(gt, gen)
        scores.append(score)

    # Add results to DataFrame
    df['SemanticSimilarity'] = scores

    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    print(f"✅ Saved comparison results to: {output_csv_path}")
    return df

# 📦 Example Usage
if __name__ == "__main__":
    input_csv = "chatbot_outputs.csv"       # Input file should have columns like: "GroundTruth", "GeneratedAnswer"
    output_csv = "semantic_similarity_results.csv"

    df_result = compare_answers_from_csv(input_csv, output_csv)
    print(df_result.head())
