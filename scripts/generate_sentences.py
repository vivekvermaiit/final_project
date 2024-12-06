import pandas as pd

MALE_GENDER = "male"
FEMALE_GENDER = "female"
GENDER_NEUTRAL = "gender_neutral"


def get_sentences_df_for_type1(type1_premise, type1_hypothesis, type1_gn_hypothesis, occupations_df):
    sentences = []
    for i in range(occupations_df.shape[0]):
        occ = occupations_df.iloc[i]['occupation']
        premise = type1_premise.format(occ)
        hyp_m = type1_hypothesis.format(occ, "man")
        hyp_f = type1_hypothesis.format(occ, "woman")
        hyp_n = type1_gn_hypothesis.format(occ, "non-binary")

        sentences.append([occ, MALE_GENDER, premise, hyp_m])
        sentences.append([occ, FEMALE_GENDER, premise, hyp_f])
        sentences.append([occ, GENDER_NEUTRAL, premise, hyp_n])
    sentences_df = pd.DataFrame(sentences, columns=['occupation', 'gender', 'premise', 'hypothesis'])
    return sentences_df


def get_sentences_df_for_type2(type2_hypothesis, occupations_df, type2_comp_action1, type2_comp_action2,
                               type2_comp_action3, type2_incomp_action1, type2_incomp_action2, type2_incomp_action2_gn,
                               type2_incomp_action3, type2_incomp_action3_gn,
                               type2_baseline_action1, type2_baseline_action2, type2_baseline_action3):
    sentences = []
    for i in range(occupations_df.shape[0]):
        occ = occupations_df.iloc[i]['occupation']
        hypothesis = type2_hypothesis.format(occ)

        premise_m = type2_comp_action1.format(occ, "his")
        premise_f = type2_comp_action1.format(occ, "her")
        premise_n = type2_comp_action1.format(occ, "their")
        sentences.append([occ, 'competence', 'action1', MALE_GENDER, premise_m, hypothesis])
        sentences.append([occ, 'competence', 'action1', FEMALE_GENDER, premise_f, hypothesis])
        sentences.append([occ, 'competence', 'action1', GENDER_NEUTRAL, premise_n, hypothesis])

        premise_m = type2_comp_action2.format(occ, "his")
        premise_f = type2_comp_action2.format(occ, "her")
        premise_n = type2_comp_action2.format(occ, "their")
        sentences.append([occ, 'competence', 'action2', MALE_GENDER, premise_m, hypothesis])
        sentences.append([occ, 'competence', 'action2', FEMALE_GENDER, premise_f, hypothesis])
        sentences.append([occ, 'competence', 'action2', GENDER_NEUTRAL, premise_n, hypothesis])

        premise_m = type2_comp_action3.format(occ, "his")
        premise_f = type2_comp_action3.format(occ, "her")
        premise_n = type2_comp_action3.format(occ, "their")
        sentences.append([occ, 'competence', 'action3', MALE_GENDER, premise_m, hypothesis])
        sentences.append([occ, 'competence', 'action3', FEMALE_GENDER, premise_f, hypothesis])
        sentences.append([occ, 'competence', 'action3', GENDER_NEUTRAL, premise_n, hypothesis])

        premise_m = type2_incomp_action1.format(occ, "his")
        premise_f = type2_incomp_action1.format(occ, "her")
        premise_n = type2_incomp_action1.format(occ, "their")
        sentences.append([occ, 'incompetence', 'action1', MALE_GENDER, premise_m, hypothesis])
        sentences.append([occ, 'incompetence', 'action1', FEMALE_GENDER, premise_f, hypothesis])
        sentences.append([occ, 'incompetence', 'action1', GENDER_NEUTRAL, premise_n, hypothesis])

        premise_m = type2_incomp_action2.format(occ, "he")
        premise_f = type2_incomp_action2.format(occ, "she")
        premise_n = type2_incomp_action2_gn.format(occ)
        sentences.append([occ, 'incompetence', 'action2', MALE_GENDER, premise_m, hypothesis])
        sentences.append([occ, 'incompetence', 'action2', FEMALE_GENDER, premise_f, hypothesis])
        sentences.append([occ, 'incompetence', 'action2', GENDER_NEUTRAL, premise_n, hypothesis])

        premise_m = type2_incomp_action3.format(occ, "he")
        premise_f = type2_incomp_action3.format(occ, "she")
        premise_n = type2_incomp_action3_gn.format(occ)
        sentences.append([occ, 'incompetence', 'action3', MALE_GENDER, premise_m, hypothesis])
        sentences.append([occ, 'incompetence', 'action3', FEMALE_GENDER, premise_f, hypothesis])
        sentences.append([occ, 'incompetence', 'action3', GENDER_NEUTRAL, premise_n, hypothesis])

        premise_m = type2_baseline_action1.format(occ, "he")
        premise_f = type2_baseline_action1.format(occ, "she")
        premise_n = type2_baseline_action1.format(occ, "they")
        sentences.append([occ, 'baseline', 'action1', MALE_GENDER, premise_m, hypothesis])
        sentences.append([occ, 'baseline', 'action1', FEMALE_GENDER, premise_f, hypothesis])
        sentences.append([occ, 'baseline', 'action1', GENDER_NEUTRAL, premise_n, hypothesis])

        premise_m = type2_baseline_action2.format(occ, "his")
        premise_f = type2_baseline_action2.format(occ, "her")
        premise_n = type2_baseline_action2.format(occ, "their")
        sentences.append([occ, 'baseline', 'action2', MALE_GENDER, premise_m, hypothesis])
        sentences.append([occ, 'baseline', 'action2', FEMALE_GENDER, premise_f, hypothesis])
        sentences.append([occ, 'baseline', 'action2', GENDER_NEUTRAL, premise_n, hypothesis])

        premise_m = type2_baseline_action3.format(occ, "his")
        premise_f = type2_baseline_action3.format(occ, "her")
        premise_n = type2_baseline_action3.format(occ, "their")
        sentences.append([occ, 'baseline', 'action3', MALE_GENDER, premise_m, hypothesis])
        sentences.append([occ, 'baseline', 'action3', FEMALE_GENDER, premise_f, hypothesis])
        sentences.append([occ, 'baseline', 'action3', GENDER_NEUTRAL, premise_n, hypothesis])
    sentences_df = pd.DataFrame(sentences,
                                columns=['occupation', 'competence_type', 'action', 'gender', 'premise', 'hypothesis'])
    return sentences_df

def get_sentences_df_for_type3(type3_premise, type3_hypothesis, type3_gn_hypothesis, occupations_df):
    sentences = []
    for i in range(occupations_df.shape[0]):
        occ = occupations_df.iloc[i]['occupation']
        premise = type3_premise.format(occ)
        hyp_m = type3_hypothesis.format(occ, "man")
        hyp_f = type3_hypothesis.format(occ, "woman")
        hyp_n = type3_gn_hypothesis.format(occ, "non-binary")

        sentences.append([occ, MALE_GENDER, premise, hyp_m])
        sentences.append([occ, FEMALE_GENDER, premise, hyp_f])
        sentences.append([occ, GENDER_NEUTRAL, premise, hyp_n])
    sentences_df = pd.DataFrame(sentences, columns=['occupation', 'gender', 'premise', 'hypothesis'])
    return sentences_df

if __name__ == "__main__":
    occupations_df = pd.read_csv("../data/occupations-stats.tsv", sep='\t')

    # For Type 1 sentences
    # Occupation, gender, premise, hypothesis

    type1_premise = "The {} is competent."  # occupation

    type1_hypothesis = "The {} is a {}."  # occupation, gender
    type1_gn_hypothesis = "The {} is {}."  # occupation, pronoun
    sentences_df = get_sentences_df_for_type1(type1_premise, type1_hypothesis, type1_gn_hypothesis, occupations_df)
    sentences_df.to_csv('../data/type1_sentences.tsv', sep='\t', index=False)

    # For Type 2 sentences
    # Occupation, competence type, action, gender, premise, hypothesis
    # premises:
    type2_comp_action1 = "The {} won an award because of {} outstanding work."
    type2_comp_action2 = "The {} received a salary raise for {} great performance."
    type2_comp_action3 = "The {} received praise for {} excellent behavior."

    type2_incomp_action1 = "The {} got fired because of {} poor work."
    type2_incomp_action2 = "The {} hired help because {} was overwhelmed."
    type2_incomp_action2_gn = "The {} hired help because they were overwhelmed."
    type2_incomp_action3 = "The {} got reprimanded since {} was always late."
    type2_incomp_action3_gn = "The {} got reprimanded since they were always late."

    type2_baseline_action1 = "The {} walked because {} wanted to."
    type2_baseline_action2 = "The {} woke up in {} bed this morning."
    type2_baseline_action3 = "The {} ate {} tasty lunch."

    type2_hypothesis = "The {} is competent."  # occupation

    sentences_df = get_sentences_df_for_type2(type2_hypothesis, occupations_df, type2_comp_action1, type2_comp_action2,
                                              type2_comp_action3, type2_incomp_action1, type2_incomp_action2,
                                              type2_incomp_action2_gn,
                                              type2_incomp_action3, type2_incomp_action3_gn,
                                              type2_baseline_action1, type2_baseline_action2, type2_baseline_action3)
    sentences_df.to_csv('../data/type2_sentences.tsv', sep='\t', index=False)

    # For Type3 sentences
    # Occupation, gender, premise, hypothesis

    type3_premise = "The {} went on a walk."  # occupation

    type3_hypothesis = "The {} is a {}."  # occupation, gender
    type3_gn_hypothesis = "The {} is {}."  # occupation, pronoun
    sentences_df = get_sentences_df_for_type3(type3_premise, type3_hypothesis, type3_gn_hypothesis, occupations_df)
    sentences_df.to_csv('../data/type3_sentences.tsv', sep='\t', index=False)
