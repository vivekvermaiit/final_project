#download dataset from https://github.com/uclanlp/corefBias/tree/master/WinoBias/wino/data
#the pro and anti equivalents mirror each other with swapped genders

import csv
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from tqdm import tqdm

# Set up NLI model
NLI_MODEL = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
TOKENIZER = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_nli(premise, hypothesis, neutral_score_bool = False):
    # Generate probabilities
    inputs = TOKENIZER.encode(premise, hypothesis, return_tensors='pt', truncation=True)
    inputs = inputs.to(DEVICE)
    NLI_MODEL.to(DEVICE)
    logits = NLI_MODEL(inputs)[0]
    if neutral_score_bool:
        probs = logits.softmax(dim = 1)

        # Parse probabilities
        entailment_prob = probs[:, 2].item()
        neutral_prob = probs[:, 1].item()
        contradiction_prob = probs[:, 0].item()

        return (entailment_prob, neutral_prob, contradiction_prob)
    
    else:
        # exclude neutral score 
        entail_contradiction_logits = logits[:,[0,2]]
        probs = entail_contradiction_logits.softmax(dim = 1)

        # Parse probabilities
        entailment_prob = probs[:, 1].item()
        contradiction_prob = probs[:, 0].item()

        return (entailment_prob, contradiction_prob)

def load_tsv_to_dataframe(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, sep='\t')
        return df
    except Exception as e:
        print(f"Error loading TSV file: {e}")
        return None
    
def assign_label(entailment, neutral, contradiction):
    """Assigns a label based on the highest probability."""
    if entailment > max(neutral, contradiction):
        return "Yes"
    elif contradiction > max(entailment, neutral):
        return "No"
    else:
        return "Neutral"
    
# Helper function to compute deltas and determine B_bergsma and B_bls
def process_group(group):
    # Compute deltas for entailment scores
    genders = group['gender'].values
    entailment_scores = group['competence_entailment_score'].values
    
    # Create gender combinations for comparison
    gender_pairs = [(i, j) for i in range(len(genders)) for j in range(i+1, len(genders))]
    delta_entailment = {f"delta_{genders[i]}_{genders[j]}": (entailment_scores[i] - entailment_scores[j]) 
                        for i, j in gender_pairs}
    
    # Find the dominant gender based on bergsma_pct_female and bls_pct_female
    dominant_bergsma_gender = "female" if group.iloc[0]['bergsma_pct_female'] > 50 else "male"
    dominant_bls_gender = "female" if group.iloc[0]['bls_pct_female'] > 50 else "male"
    
    # Filter out gender_neutral rows for max gender calculation
    filtered_group = group[group['gender'].isin(['male', 'female'])]
    if not filtered_group.empty:
        max_entailment_index = filtered_group['competence_entailment_score'].idxmax()
        max_gender = group.loc[max_entailment_index, 'gender']
    else:
        max_gender = None  # If no male or female rows are present, this will be None
    
    # Compute B_bergsma and B_bls only if max_gender is determined
    B_bergsma = max_gender == dominant_bergsma_gender if max_gender else False
    B_bls = max_gender == dominant_bls_gender if max_gender else False
    
    # Add results to each row in the group
    for col, delta in delta_entailment.items():
        group[col] = delta
    group['B_bergsma'] = B_bergsma
    group['B_bls'] = B_bls
    
    return group

# Function to calculate and print proportions for delta and B columns automatically
def calculate_proportions_auto(df):
    # Identify delta columns
    delta_columns = [col for col in df.columns if col.startswith("delta_")]
    
    # Identify B columns
    bias_columns = [col for col in df.columns if col.startswith("B_")]
    
    # Calculate proportions for delta columns
    for col in delta_columns:
        positive_count = (df[col] > 0).sum()
        negative_count = (df[col] < 0).sum()
        total_count = len(df[col])
        print(f"Column: {col}")
        print(f"  Positive Proportion: {positive_count / total_count:.2%}")
        print(f"  Negative Proportion: {negative_count / total_count:.2%}")
    
    # Calculate proportions for bias columns
    for col in bias_columns:
        true_count = (df[col] == True).sum()
        false_count = (df[col] == False).sum()
        total_count = len(df[col])
        print(f"Column: {col}")
        print(f"  True Proportion: {true_count / total_count:.2%}")
        print(f"  False Proportion: {false_count / total_count:.2%}")

    
occupation_df = load_tsv_to_dataframe("./data/occupations-stats.tsv")
type1_premises_df = load_tsv_to_dataframe("./data/type1_premises.tsv")
type2_hypothesis_df = load_tsv_to_dataframe("./data/type2_hypothesis.tsv")
print(type1_premises_df.head())

#add occupation stats to rows in dataset (if they're not already there)
if 'bergsma_pct_female' not in type1_premises_df.columns and 'bls_pct_female' not in type1_premises_df.columns:
    type1_premises_df = type1_premises_df.merge(occupation_df[['occupation', 'bergsma_pct_female', 'bls_pct_female']], 
                                                on='occupation', 
                                                how='left')
if 'bergsma_pct_female' not in type2_hypothesis_df.columns and 'bls_pct_female' not in type2_hypothesis_df.columns:
    type2_hypothesis_df = type2_hypothesis_df.merge(occupation_df[['occupation', 'bergsma_pct_female', 'bls_pct_female']], 
                                                on='occupation', 
                                                how='left')


type1_premises_df['competence_hypothesis'] = type1_premises_df['occupation'].apply(lambda p : "the [PROFESSION_NAME] is competent.".replace("[PROFESSION_NAME]", p))
type2_hypothesis_df['adequacy_premise'] = type2_hypothesis_df['occupation'].apply(lambda p : "the [PROFESSION_NAME] is adequate.".replace("[PROFESSION_NAME]", p))
type2_hypothesis_df['competence_premise'] = type2_hypothesis_df['occupation'].apply(lambda p : "the [PROFESSION_NAME] is competent.".replace("[PROFESSION_NAME]", p))

recompute = True

if recompute:

    for i, row in tqdm(type1_premises_df.iterrows(), total=len(type1_premises_df), desc="Running type 1 experiment"):
        competence_entailment_score, competence_neutral_score, competence_contradiction_score = run_nli(row['sentence'], row['competence_hypothesis'], neutral_score_bool=True)
        type1_premises_df.at[i, 'competence_entailment_score'] = competence_entailment_score
        type1_premises_df.at[i, 'competence_neutral_score'] = competence_neutral_score
        type1_premises_df.at[i, 'competence_contradiction_score'] = competence_contradiction_score
        competence_label = assign_label(competence_entailment_score, competence_neutral_score, competence_contradiction_score)
        type1_premises_df.at[i, 'competence_label'] = competence_label

    # Group by occupation and baseline, then process each group
    type1_premises_df = type1_premises_df.groupby(['occupation', 'baseline']).apply(process_group)

    # Reset index for a clean DataFrame
    type1_premises_df.reset_index(drop=True, inplace=True)

    type1_premises_df.to_csv('./data/type1_premises.tsv',sep='\t',index=False)

    for i, row in tqdm(type2_hypothesis_df.iterrows(), total=len(type2_hypothesis_df), desc="Running type 2 experiment"):
        adequacy_entailment_score, adequacy_neutral_score, adequacy_contradiction_score = run_nli(row['adequacy_premise'], row['sentence'], neutral_score_bool=True)
        type2_hypothesis_df.at[i, 'adequacy_entailment_score'] = adequacy_entailment_score
        type2_hypothesis_df.at[i, 'adequacy_neutral_score'] = adequacy_neutral_score
        type2_hypothesis_df.at[i, 'adequacy_contradiction_score'] = adequacy_contradiction_score
        adequacy_label = assign_label(adequacy_entailment_score, adequacy_neutral_score, adequacy_contradiction_score)
        type2_hypothesis_df.at[i, 'adequacy_label'] = adequacy_label

        competence_entailment_score, competence_neutral_score, competence_contradiction_score = run_nli(row['competence_premise'], row['sentence'], neutral_score_bool=True)
        type2_hypothesis_df.at[i, 'competence_entailment_score'] = competence_entailment_score
        type2_hypothesis_df.at[i, 'competence_neutral_score'] = competence_neutral_score
        type2_hypothesis_df.at[i, 'competence_contradiction_score'] = competence_contradiction_score
        competence_label = assign_label(competence_entailment_score, competence_neutral_score, competence_contradiction_score)
        type2_hypothesis_df.at[i, 'competence_label'] = competence_label

    # Group by occupation and baseline, then process each group
    type2_hypothesis_df = type2_hypothesis_df.groupby(['occupation', 'baseline']).apply(process_group)

    # Reset index for a clean DataFrame
    type2_hypothesis_df.reset_index(drop=True, inplace=True)

    type2_hypothesis_df.to_csv('./data/type2_hypothesis.tsv',sep='\t',index=False)


# Calculate proportions for type1_premises_df
print("Stats for type1_premises_df:")
calculate_proportions_auto(type1_premises_df)

# Calculate proportions for type2_hypothesis_df
print("\Stats for type2_hypothesis_df:")
calculate_proportions_auto(type2_hypothesis_df)