import mwapi
import re
import pickle
import numpy as np
import nltk
import mwparserfromhell

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import sent_tokenize


from keras import backend as K
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=10, inter_op_parallelism_threads=10)))

nltk.download('punkt')  # fetching pretrained PunktSentenceTokenizer model


def declutter(obj):
    """
    function for removing elements from the wikitext that would otherwise not be removed fully by using the
    strip_code() function
    :param obj: Wikicode object from get_sections()
    :return: Wikicode stripped of all headings, tags (table was the tricky one bringing problems) and thumbnail
    wikilinks, and external url links, concatenates all the stripped statements into a single string
    """

    outstring = ''
    for element in obj:
        headings = element.filter_headings()
        for heading in headings:
            element.remove(heading)

        table_tags = element.filter_tags(element.RECURSE_OTHERS, matches=lambda n: n.tag == "table")
        for tag in table_tags:
            element.remove(tag)

        thumbnail_wikilinks = element.filter_wikilinks(element.RECURSE_OTHERS, matches="thumb")
        for link in thumbnail_wikilinks:
            element.remove(link)

        external_links = element.filter_external_links(element.RECURSE_OTHERS)
        for exlink in external_links:
            element.remove(exlink)

        elements = element.strip_code().split('\n')
        # print(elements)
        for e in elements:
            # some list entries are empty strings or strings with one character
            re.sub(r"^\;\s*$", "", e)
            re.sub(r"^\s+$", "", e)
            if e == '':
                elements.remove(e)

        outstring = '\n'.join(elements)
    return outstring


def declutter_header(obj):
    """
    function for removing elements from the wikitext (leading section in particular)that would otherwise not be
    removed fully by using the strip_code() function
    :param obj: Wikicode object from get_sections()
    :return: Wikicode stripped of all headings, tags (table was the tricky one bringing problems) and thumbnail
    wikilinks, and external url links, concatenates all the stripped statements into a single string
    """

    headings = obj.filter_headings()
    for heading in headings:
        obj.remove(heading)

    table_tags = obj.filter_tags(obj.RECURSE_OTHERS, matches=lambda n: n.tag == "table")
    for tag in table_tags:
        obj.remove(tag)

    thumbnail_wikilinks = obj.filter_wikilinks(obj.RECURSE_OTHERS, matches="thumb")
    for link in thumbnail_wikilinks:
        obj.remove(link)

    external_links = obj.filter_external_links(obj.RECURSE_OTHERS)
    for exlink in external_links:
        obj.remove(exlink)

    elements = obj.strip_code().split('\n')
    # print(elements)
    for e in elements:
        # some list entries are empty strings or strings with one character
        re.sub(r"^\;\s*$", "", e)
        re.sub(r"^\s+$", "", e)
        if e == '':
            elements.remove(e)

    return '\n'.join(elements)


def construct_test_data(wiki_article_title, section_title, statements):
    """
    function for appending the data into one list for feeding to the model
    :param wiki_article_title: Wikipedia article title
    :param section_title: Section title
    :param statements: statements in the section
    :return: list
    """
    data = []
    for sentence in sent_tokenize(statements):
        data.append([wiki_article_title, section_title, sentence])
    return data

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
    titles = session.get(
        action="query",
        list="search",
        format="json",
        srsearch='intitle:' + title_query,
    )

    sample_data = []  # creating the dataset

    # conditional statement to handle empty response
    if titles['query']['searchinfo']['totalhits'] == 0:
        while titles['query']['searchinfo']['totalhits'] == 0:
            title_query = input('Title not found. Kindly check if you typed it correctly: (Do Ctrl + C to exit)\n>')
            response = session.get(
                action="query",
                list="search",
                format="json",
                srsearch='intitle:' + title_query,  # nifty workaround the search-title-disabled error
            )
            if titles['query']['searchinfo']['totalhits'] > 0:
                break

    for item in titles['query']['search']:
        # checks if title is actually in the title text and handling misspellings by checking if there is a 'suggested'
        # key in 'searchinfo'
        if all([title_query.lower() not in item['title'].lower(),
                'suggestion' not in titles['query']['searchinfo'].keys()]):
            continue

        # GET request to search for a title
        response = session.get(
            action="parse",
            format="json",
            prop="wikitext",  # getting the wikitext property to parse
            pageid=int("{pageid}".format(**item)),
        )

        wikitext = mwparserfromhell.parse(response['parse']['wikitext']["*"])  # creating the wikicode object
        wiki_article_title = item['title']  # extracting the wikipedia article for later use
        lead_section_title = "MAIN_TITLE"

        # get_sections returns a list of content per section. with include_lead=True,
        # the first item in this lis is the lead section content
        lead_section_text = wikitext.get_sections(include_lead=True, include_headings=False)[0]
        lead_section_text = declutter_header(lead_section_text)

        # extending the content into the dataset list
        sample_data.extend(construct_test_data(wiki_article_title, lead_section_title, lead_section_text))

        # getting all the level 1 headings
        sections = wikitext.filter_headings(matches=lambda n: n.level == 2)
        exceptions = ['See also', 'References', 'External links', 'Further reading',
                      'Footnotes', 'Notes', 'Bibliography', 'Literature']

        for section in sections:
            # getting rid of unwanted sections, the re sub is for removing trailing and leading whitespaces in the
            # section titles
            if re.sub(r"^\s+|\s+$", "", section.title.strip_code()) not in exceptions:
                section_title = section.title.strip_code()
                section_text = wikitext.get_sections(matches=section_title, include_headings=False)
                section_text = declutter(section_text)

                sample_data.extend(construct_test_data(wiki_article_title, section_title, section_text))


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
    output.sort(key=lambda x: x[2])  # sorting by order of prediction score
    print('Wiki Title\tText\tPrediction\n')
    for result in output:
        print(result[0]+'\t'+result[1]+'\t'+str(result[2]))

