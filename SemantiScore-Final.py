#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install sentence-transformers pandas


# In[2]:


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
    print(f"Saved comparison results to: {output_csv_path}")
    return df


# In[5]:


#changing the encoding of input file
#run this only if required
import chardet

file_path = r'D:\UTAustin\AI_in_healthcare\Summer_2025_Manasa Koppula\Project\GenerateAnswers.csv'
# Read a portion of the file to detect encoding
with open(file_path, 'rb') as f:
    raw_data = f.read()
    result = chardet.detect(raw_data)

detected_encoding = result['encoding']
print(f"Detected encoding: {detected_encoding}") 

input_path = file_path
output_path = r'D:\UTAustin\AI_in_healthcare\Summer_2025_Manasa Koppula\Project\gptGeneratedAnswers.csv'

# Read with detected encoding, write as UTF-8
with open(input_path, 'r', encoding=detected_encoding, errors='ignore') as source_file:
    content = source_file.read()

with open(output_path, 'w', encoding='utf-8') as target_file:
    target_file.write(content)


# In[7]:


if __name__ == "__main__":
    input_csv = r'D:\UTAustin\AI_in_healthcare\Summer_2025_Manasa Koppula\Project\gptGeneratedAnswers.csv'  # Input file should have columns like: "GroundTruth", "GeneratedAnswer"
    output_csv = r'D:\UTAustin\AI_in_healthcare\Summer_2025_Manasa Koppula\Project\semantic_similarity_results.csv'

    df_result = compare_answers_from_csv(input_csv, output_csv)
    print(df_result.head())


# In[ ]:




