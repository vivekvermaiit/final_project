import pandas as pd

MALE_GENDER = "male"
FEMALE_GENDER = "female"
GENDER_NEUTRAL = "gender_neutral"


def create_sentences_df(template, template_gn, df):
    sentences = []
    for i in range(df.shape[0]):
        occ = df.iloc[i]['occupation']
        sentence_m = template.format(occ, "he")
        sentence_f = template.format(occ, "she")
        sentence_n = template_gn.format(occ, "they")
        sentences.append([occ, MALE_GENDER, sentence_m])
        sentences.append([occ, FEMALE_GENDER, sentence_f])
        sentences.append([occ, GENDER_NEUTRAL, sentence_n])
    sentences_df = pd.DataFrame(sentences, columns=['occupation', 'gender', 'sentence'])
    return sentences_df


def store_sentences_df(sentences_df, path):
    sentences_df.to_csv(path, sep='\t', index=False)


if __name__ == "__main__":
    df = pd.read_csv("../data/occupations-stats.tsv", sep='\t')
    template = "the {} hired help because {} was overwhelmed"  # occupation, pronoun
    template_gn = "the {} hired help because {} were overwhelmed"
    sentences_df = create_sentences_df(template, template_gn, df)
    store_sentences_df(sentences_df, '../data/all_sentences.tsv')
