# Classifying Statements Within an Article 
*Barak Achillah Asidi*

## Requirements
1. `mwapi`
2. `keras==2.1.5`
3. `tensorflow==1.7.0`
4. `pandas`
5. `h5py`
6. `scikit-learn==0.18.1`
77. `beautifulsoup4`

Note: `mwapi.Session` throws an error when using Python 2.7 (even though the package can be downloaded). Using Python 3.x recommended. I used Python 3.6.

## Task Description
Program that:
1. Receives as input the title of a English Wikipedia article.
2. Retrieves the text of that article from the MediaWiki API.
3. Identifies individual sentences within that text, along with the corresponding section titles.
4. Runs those sentences through the model to classify them.
5. Outputs the sentences, one per line, sorted by score given by the model.

