import mwapi
import re
import pickle
import numpy as np
import nltk
nltk.download('punkt')  # fetching pretrained PunktSentenceTokenizer model


from keras.models import load_model
from bs4 import BeautifulSoup
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import sent_tokenize


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
    vocab_w2v = pickle.load(open('embeddings/word_dict_en.pck', 'rb'), encoding='latin1')

    # load the section dictionary.
    section_dict = pickle.load(open('embeddings/section_dict_en.pck', 'rb'), encoding='latin1')


    # construct the training data
    X = []
    sections = []
    outstring=[]
    for row in statements:
        try:
            print(row)
            statement_text = text_to_word_list(row[2])

            X_inst = []
            for word in statement_text:

                if max_len != -1 and len(X_inst) >= max_len:
                    continue
                if word not in vocab_w2v:
                    X_inst.append(vocab_w2v['UNK'])
                else:
                    X_inst.append(vocab_w2v[word])

            # extract the section, and in case the section does not exist in the model, then assign UNK
            section = row[1].lower()
            sections.append(np.array([section_dict[section] if section in section_dict else 0]))

            X.append(X_inst)
            outstring.append(str(row[2]))

        except Exception as e:
            print(row)
            print(e)
    X = pad_sequences(X, maxlen=max_len, value=vocab_w2v['UNK'], padding='pre')

    return X, np.array(sections), outstring



if __name__ == '__main__':

    title_query = input('Type in the title of the article: \n> ')  # query for the Wikipedia title
    session = mwapi.Session('https://en.wikipedia.org')  # creating a new session

    # GET request to search for a title
    response = session.get(
        action="query",
        list="search",
        format="json",
        srsearch=title_query,
        # srwhat='title', search-title-disabled
    )

    sample_data = []
    footnote_tags = '\[\d+\]'  # regex pattern or removing footnotes i.e. [1]

    for item in response['query']['search']:
        # going through all search responses
        content = session.get(
            action="parse",
            pageid=int("{pageid}".format(**item)),  # getting the content as specified by page id
            format="json"
        )
        wiki_article_title = item['title']
        # Introductory section that is not in the table of contents
        section_title = 'MAIN_SECTION'  # using this label because it was used in the sample text file
        soup = BeautifulSoup(content['parse']['text']['*'], 'html.parser')
        start = soup.find_all('table')  # the preliminary section starts after the table tag
        try:
            statements=''
            for element in start[0].next_siblings:
                    # finding all p tags and ending at the next div tag
                    if element.name == 'p':
                        statements = element.get_text()
                        statements = re.sub(footnote_tags, '', statements)  # removing the footnote tags
                        statements = sent_tokenize(statements)  # splitting it up into individual sentences

                        for statement in statements:
                            if statement is not '':
                                sample_data.append([wiki_article_title, section_title, statement])

                    if element.name == 'div':
                        break


        except IndexError:
                    continue

        exceptions = ['See also', 'References',
                                 'External links','Further reading']
        for k in content['parse']['sections']:
            # Looping through all the sections
            if k['line'] not in exceptions:  # filtering out sections with the following titles
                if k['toclevel'] is 1:  # specifying only level 1 sections
                    section_content = session.get(
                        action="parse",
                        pageid=int("{pageid}".format(**item)),
                        section=(k['number']),  # filtering by section number
                        format="json"
                    )
                    section_title = k['line']
                    soup = BeautifulSoup(section_content['parse']['text']['*'], 'html.parser')
                    statements=''
                    for x in soup.find_all('p'):
                        # finding all <p> tags
                        statements = x.text
                        statements = re.sub(footnote_tags, '', statements)  # removing the footnote tags
                        statements = sent_tokenize(statements)  # splitting it up into individual sentences

                        for statement in statements:
                            if statement is not '':
                                sample_data.append([wiki_article_title, section_title, statement])

    # load the model
    model = load_model('model/fa_en_model_rnn_attention_section.h5')

    # load the data
    max_seq_length = model.input[0].shape[1].value

    X, sections, outstring = construct_instance_reasons(sample_data, max_seq_length)

    # classify the data
    pred = model.predict([X, sections])
    output = []

    # adding results to a list so as to sort it
    for idx, y_pred in enumerate(pred):
        output.append([sample_data[idx][0],outstring[idx],y_pred[0]])

    # printing out results to console
    output.sort(key=lambda x: x[2]) # sorting by order of prediction score
    print('Wiki Title\tText\tPrediction\n')
    for result in output:
        print(result[0]+'\t'+result[1]+'\t'+str(result[2]))

