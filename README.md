# Classifying Statements Within an Article 
*Barak Achillah Asidi*

## Requirements
1. `mwapi` - querying and parsing wikimedia content
2. `keras` - running the model
3. `tensorflow` - running the model
5. `nltk` - tokenizing the paragrapgh into sentences
6. `beautifulsoup4` - parsing the HTML content

Note: `mwapi.Session` throws an error when using Python 2.7 (even though the package can be downloaded). Using Python 3.x recommended.

## Task Description
Program that:
1. Receives as input the title of a English Wikipedia article.
2. Retrieves the text of that article from the MediaWiki API.
3. Identifies individual sentences within that text, along with the corresponding section titles.
4. Runs those sentences through the model to classify them.
5. Outputs the sentences, one per line, sorted by score given by the model.

## Running the Application
1. Create a new virtual environment in Python 3.x using `python3 -m venv env`
2. Initialise into your virtual environment using `source env/bin/activate`
2. Install needed libraries using `pip install -r requirements.txt`
4. Run the program.

## Notes
 - The suggested `mwparserfromhell` was bringing problems during parsing of the HTML page, so I opted for BeautifulSoup
 - Since the models and the pickled files were created using Python 2 and mwapi only works in Python 3.x these files had to be ported with some alterations due to the unicode compatibility issues with the pickle functions. Tensorflow throws some warnings about depreciation of some functions.
 - Since the title query would bring mutiple results, I thought it apt to include a column for the article title the sentence comes from.
 - if the 'punkd' file (for paragraph tokenization) cannot be installed due to an SSL verification error, you have to install the SSL certificates. The install file should be found in the Python 3.x directory (for Mac, it was a 'Install Certificates.command' file).
 - For some reason, the output in my console showed duplicates lines
 - The files needed to run the program (that are referenced) are [`word_dict_en.pck`](https://drive.google.com/drive/folders/1dlocPHPz6Giv9nS8rR4t6kes8nlJ3inX?usp=sharing) , [`section_dict_en.pck`](https://drive.google.com/drive/folders/1dlocPHPz6Giv9nS8rR4t6kes8nlJ3inX?usp=sharing) and [`fa_en_model_rnn_attention_section.h5`](https://drive.google.com/drive/folders/166ok0FmW-SiMNJl9ZYpeVjc8BeO1195W?usp=sharing)
  
 