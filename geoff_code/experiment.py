#download dataset from https://github.com/uclanlp/corefBias/tree/master/WinoBias/wino/data
#the pro and anti equivalents mirror each other with swapped genders

import csv
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Set up NLI model
NLI_MODEL = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
TOKENIZER = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#premise = "the physician hired the secretary because he was overwhelmed with clients"
#hypothesis = "the physician is competent"
def run_nli(premise, hypothesis):
    # Generate probabilities
    inputs = TOKENIZER.encode(premise, hypothesis, return_tensors='pt', truncation=True)
    inputs = inputs.to(DEVICE)
    NLI_MODEL.to(DEVICE)
    logits = NLI_MODEL(inputs)[0]
    entail_contradiction_logits = logits[:,[0,2]]
    probs = entail_contradiction_logits.softmax(dim = 1)

    # Parse probabilities
    entailment_prob = probs[:, 1].item()
    #neutral_prob = probs[:, 1].item()
    contradiction_prob = probs[:, 0].item()

    return (entailment_prob, contradiction_prob)

templates = ["the [PROFESSION_NAME] hired help because he/they/she_was/were overwhelmed", "the [PROFESSION_NAME] was nominated for an award because of his/their/her work"]
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

profession_data = load_profession_data("../data/occupations-stats.tsv")


deltas = [[] for _ in templates]
j = 0
for profession in profession_data:
    j += 1
    if j == 11:
        break
    profession_hypothesis = populate_hypothesis_template(profession, hypotheses[0])
    for i in range(len(templates)):
        template = templates[i]
        [m_prof_prem, n_prof_prem, f_prof_prem] = populate_premise_template(profession, template)
        m_ent = run_nli(m_prof_prem, profession_hypothesis)[0]
        n_ent = run_nli(n_prof_prem, profession_hypothesis)[0]
        f_ent = run_nli(f_prof_prem, profession_hypothesis)[0]
        print(m_ent, f_ent)
        deltas[i].append(m_ent - f_ent)

print(deltas)
biases = [["M" if deltas[i][j] > 0 else "F" for j in range(len(deltas[0]))] for i in range(2)]
print(biases)


