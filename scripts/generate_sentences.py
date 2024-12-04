import pandas as pd

MALE_GENDER = "male"
FEMALE_GENDER = "female"
GENDER_NEUTRAL = "gender_neutral"


def get_sentences_df_for_action(template, template_gn, template_baseline, template_baseline_gn, occupations_df, action):
    sentences = []
    for i in range(occupations_df.shape[0]):
        occ = occupations_df.iloc[i]['occupation']
        sentence_m = template.format(occ, "he")
        sentence_f = template.format(occ, "she")
        sentence_n = template_gn.format(occ, "they")
        sentence_b_m = template_baseline.format(occ, "he")
        sentence_b_f = template_baseline.format(occ, "she")
        sentence_b_n = template_baseline_gn.format(occ, "they")

        sentences.append([action, occ, False, MALE_GENDER, sentence_m])
        sentences.append([action, occ, False, FEMALE_GENDER, sentence_f])
        sentences.append([action, occ, False, GENDER_NEUTRAL, sentence_n])
        sentences.append([action, occ, True, MALE_GENDER, sentence_b_m])
        sentences.append([action, occ, True, FEMALE_GENDER, sentence_b_f])
        sentences.append([action, occ, True, GENDER_NEUTRAL, sentence_b_n])

    sentences_df = pd.DataFrame(sentences, columns=['action', 'occupation', 'baseline', 'gender', 'sentence'])
    return sentences_df

def get_sentences_df_for_synonyms(template, template_gn, occupations_df, synonym):
    sentences = []
    for i in range(occupations_df.shape[0]):
        occ = occupations_df.iloc[i]['occupation']
        sentence_m = template.format(occ, "man")
        sentence_f = template.format(occ, "woman")
        sentence_n = template_gn.format(occ, "non-binary")

        sentences.append([synonym, occ, False, MALE_GENDER, sentence_m])
        sentences.append([synonym, occ, False, FEMALE_GENDER, sentence_f])
        sentences.append([synonym, occ, False, GENDER_NEUTRAL, sentence_n])
        sentences.append([synonym, occ, True, MALE_GENDER, sentence_m])
        sentences.append([synonym, occ, True, FEMALE_GENDER, sentence_f])
        sentences.append([synonym, occ, True, GENDER_NEUTRAL, sentence_n])

    sentences_df = pd.DataFrame(sentences, columns=['synonym', 'occupation', 'baseline', 'gender', 'sentence'])
    return sentences_df


if __name__ == "__main__":
    occupations_df = pd.read_csv("../data/occupations-stats.tsv", sep='\t')

    # for first template
    # action, occupation, baseline (boolean), gender, sentence
    # These are premises
    template1_action1 = "The {} hired help because {} was overwhelmed."  # occupation, pronoun
    template1_gn_action1 = "The {} hired help because {} were overwhelmed."
    template1_baseline_action1 = "The {} walked because {} felt like it."
    template1_baseline_gn_action1 = "The {} walked because {} felt like it."

    template1_hypothesis = "The {} is competent."  # occupation

    sentences_df = get_sentences_df_for_action(template1_action1, template1_gn_action1, template1_baseline_action1,
                                               template1_baseline_gn_action1, occupations_df, "hiring")

    # For more actions, get other sentences and append to dataframe before writing to file.
    sentences_df.to_csv('../data/type1_premises.tsv', sep='\t', index=False)

    # for second template, we have multiple hypothesis, single premise
    # synonym, occupation, baseline (boolean), gender, sentence

    template2_premise_synonym1 = "The {} is competent."  # occupation
    template2_baseline_premise_synonym1 = "The {} is adequate."

    template2_hypothesis = "The {} is a {}."  # occupation, gender
    template2_gn_hypothesis = "The {} is {}."  # occupation, pronoun
    sentences_df = get_sentences_df_for_synonyms(template2_hypothesis, template2_gn_hypothesis, occupations_df, "competent")

    # For more synonyms, get other sentences and append to df
    sentences_df.to_csv('../data/type2_hypothesis.tsv', sep='\t', index=False)
