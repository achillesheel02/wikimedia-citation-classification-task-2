# import statements
import mwapi
import argparse
import re
import pickle
import numpy as np
import types

from keras.models import load_model
from bs4 import BeautifulSoup
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer

from keras.utils import to_categorical

from keras import backend as K
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=10, inter_op_parallelism_threads=10)))


def text_to_word_list(text):
    # check first if the statements is longer than a single sentence.
    sentences = re.compile('\.\s+').split(str(text))
    if len(sentences) != 1:
        # text = sentences[random.randint(0, len(sentences) - 1)]
        text = sentences[0]

    text = str(text).lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.strip().split()

    return text

def construct_instance_reasons(statements,  max_len=-1):
    # Load the vocabulary
    vocab_w2v = pickle.load(open('embeddings/word_dict_en.pck', 'rb'),encoding='latin1')

    # load the section dictionary.
    section_dict = pickle.load(open('embeddings/section_dict_en.pck', 'rb'),encoding='latin1')


    # construct the training data
    X = []
    sections = []
    y = ['True','False']
    outstring=[]
    for row in statements:
        try:
            print(row)
            statement_text = text_to_word_list(row[1])

            X_inst = []
            for word in statement_text:

                if max_len != -1 and len(X_inst) >= max_len:
                    continue
                if word not in vocab_w2v:
                    X_inst.append(vocab_w2v['UNK'])
                else:
                    X_inst.append(vocab_w2v[word])

            # extract the section, and in case the section does not exist in the model, then assign UNK
            section = row[0].lower()
            sections.append(np.array([section_dict[section] if section in section_dict else 0]))

            label = [True, False]

            # some of the rows are corrupt, thus, we need to check if the labels are actually boolean.
            if type(label) != bool:
                continue


            X.append(X_inst)
            outstring.append(str(row[1]))
            #entity_id  revision_id timestamp   entity_title    section_id  section prg_idx sentence_idx    statement   citations

        except Exception as e:
            print (row)
            print (e)
    X = pad_sequences(X, maxlen=max_len, value=vocab_w2v['UNK'], padding='pre')

    encoder = LabelBinarizer()
    y = encoder.fit_transform(y)
    y = to_categorical(y)

    return X, np.array(sections), y, encoder, outstring



if __name__ == '__main__':
    p = input()

    title_query = p  # query for the Wikipedia title
    session = mwapi.Session('https://en.wikipedia.org')  # creating a new session

    # GET request to search for a title
    response = session.get(
        action="query",
        list="search",
        format="json",
        srsearch=title_query

    )

    footnote_tags = '\[\d+\]|(\n)'  # regex pattern or removing footnotes i.e. [1]
    sample_data = []

    for item in response['query']['search']:
        # going through all search responses
        content = session.get(
            action="parse",
            pageid=int("{pageid}".format(**item)),  # getting the content as specified by page id
            format="json"
        )

        # there's a section that is before the first section that cannot be parsed
        section_title = 'MAIN_SECTION'  # using this label because it was used in the sample text file
        soup = BeautifulSoup(content['parse']['text']['*'], 'html.parser')
        start = soup.find_all('table',
                              class_="infobox vcard")  # the section starts after <table class='infobox vcard'>...</table>
        try:
            for element in start[0].next_siblings:

                    # finding all p tags and ending at the next div tag
                    if element.name == 'p':
                        statements = re.sub(footnote_tags, '', element.get_text())  # removing the footnote tags
                        statements = statements.split(". ")  # splitting it up into individual sentences

                        for statement in statements:
                            if statement is not '':
                                statement.replace("\\\\'", "'")  # replacing \' with '
                                sample_data.append([section_title, statement])

                    if element.name == 'div':
                        break
        except IndexError:
                    continue


        for k in content['parse']['sections']:
            # Looping through all the sections
            if k['line'] not in ['See also', 'References',
                                 'External links']:  # filtering out sections with the followwing titles
                if k['toclevel'] is 1:  # specifying only level 1 sections
                    section_content = session.get(
                        action="parse",
                        pageid=int("{pageid}".format(**item)),
                        section=(k['number']),  # filtering by section number
                        format="json"
                    )
                    section_title = k['line']
                    soup = BeautifulSoup(section_content['parse']['text']['*'], 'html.parser')
                    label=''
                    for x in soup.find_all('p'):
                        # finding all <p> tags
                        statements = re.sub(footnote_tags, '', x.text)  # removing the footnote tags
                        statements = statements.split(". ")  # splitting it up into individual sentences

                        for statement in statements:
                            if statement is not '':
                                statement.replace("\\\\'", "'")  # replacing \' with '
                                sample_data.append([section_title, statement])
        print(sample_data)
    # load the model
    model = load_model('model/fa_en_model_rnn_attention_section.h5')

    # load the data
    max_seq_length = model.input[0].shape[1].value

    X, sections, y, encoder, outstring = construct_instance_reasons(sample_data, max_seq_length)
    print(X)
    print(sections)
    # classify the data
    pred = model.predict([X, sections])
    print(pred)
    # store the predictions: printing out the sentence text, the prediction score, and original citation label.
    outstr = 'Text\tPrediction\tCitation\n'
    for idx, y_pred in enumerate(pred):
        outstr += outstring[idx] + '\t' + str(y_pred) + '\n'

    fout = open('output_predictions_sections.tsv', 'wt')
    print(outstr)
    fout.write(outstr)
    fout.flush()
    fout.close()
