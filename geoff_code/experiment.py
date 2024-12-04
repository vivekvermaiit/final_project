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

#premise = "the physician hired the secretary because he was overwhelmed with clients"
#hypothesis = "the physician is competent"
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
        entail_contradiction_logits = logits[:,[0,2]]
        probs = entail_contradiction_logits.softmax(dim = 1)

        # Parse probabilities
        entailment_prob = probs[:, 1].item()
        #neutral_prob = probs[:, 1].item()
        contradiction_prob = probs[:, 0].item()

        return (entailment_prob, contradiction_prob)

'''templates = ["the [PROFESSION_NAME] hired help because he/they/she_was/were overwhelmed", "the [PROFESSION_NAME] was nominated for an award because of his/their/her work"]
hypotheses = ["the [PROFESSION_NAME] is competent"]

def populate_premise_template(profession_name, template):
    male_sentence = template.replace("[PROFESSION_NAME]", profession_name).replace("he/they/she_was/were", "he was").replace("his/their/her", "his")
    neutral_sentence = template.replace("[PROFESSION_NAME]", profession_name).replace("he/they/she_was/were", "they were").replace("his/their/her", "their")
    female_sentence = template.replace("[PROFESSION_NAME]", profession_name).replace("he/they/she_was/were", "she was").replace("his/their/her", "her")
    return [male_sentence, neutral_sentence, female_sentence]

def populate_hypothesis_template(profession_name, hypothesis):
    hypothesis_populated = hypothesis.replace("[PROFESSION_NAME]", profession_name)
    return hypothesis_populated


def load_profession_data(file_path):
    #returns dict with key : value format being occupation : (bergsma_pct_female, bls_pct_female)
    profession_data = {}
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            profession = row['occupation']
            bersma_pct_female = float(row['bergsma_pct_female'])
            bls_pct_female = float(row['bls_pct_female'])
            profession_data[profession] = (bersma_pct_female, bls_pct_female)
    return profession_data

profession_data = load_profession_data("../data/occupations-stats.tsv")'''

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
    delta_entailment = {f"delta_{genders[i]}_{genders[j]}": abs(entailment_scores[i] - entailment_scores[j]) 
                        for i, j in gender_pairs}
    
    # Find the dominant gender based on bergsma_pct_female and bls_pct_female
    dominant_bergsma_gender = "female" if group.iloc[0]['bergsma_pct_female'] > 50 else "male"
    dominant_bls_gender = "female" if group.iloc[0]['bls_pct_female'] > 50 else "male"
    
    # Identify the row with the highest entailment score
    max_entailment_index = entailment_scores.argmax()
    max_gender = genders[max_entailment_index]
    
    # Compute B_bergsma and B_bls
    B_bergsma = max_gender == dominant_bergsma_gender
    B_bls = max_gender == dominant_bls_gender
    
    # Add results to each row in the group
    for col, delta in delta_entailment.items():
        group[col] = delta
    group['B_bergsma'] = B_bergsma
    group['B_bls'] = B_bls
    
    return group
    

occupation_df = load_tsv_to_dataframe("./data/occupations-stats.tsv")
type1_premises_df = load_tsv_to_dataframe("./data/type1_premises.tsv")
type2_hypothesis_df = load_tsv_to_dataframe("./data/type2_hypothesis.tsv")
print(type1_premises_df.head())

#add occupation stats to rows in dataset
type1_premises_df = type1_premises_df.merge(occupation_df[['occupation', 'bergsma_pct_female', 'bls_pct_female']], 
                                            on='occupation', 
                                            how='left')
print(type1_premises_df.head())



type1_premises_df['competence_hypothesis'] = type1_premises_df['occupation'].apply(lambda p : "the [PROFESSION_NAME] is competent.".replace("[PROFESSION_NAME]", p))
type2_hypothesis_df['adequacy_premise'] = type2_hypothesis_df['occupation'].apply(lambda p : "the [PROFESSION_NAME] is adequate.".replace("[PROFESSION_NAME]", p))
type2_hypothesis_df['competence_premise'] = type2_hypothesis_df['occupation'].apply(lambda p : "the [PROFESSION_NAME] is competent.".replace("[PROFESSION_NAME]", p))

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

type2_hypothesis_df.to_csv('./data/type2_hypothesis.tsv',sep='\t',index=False)
