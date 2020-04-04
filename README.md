# Disaster_Response
Message classification app using natural language processing - Udacity Project

Udacity - Data Scientist Nanodegree [Project - Disaster Response Pipelines]

Link to the Application [here](https://responsedisaster-rpdieego.herokuapp.com/)

>Following a disaster, tipically you will get millions and millions of communications, either direct of via social media right at the time when disaster response organizations have the least capacity to filter and then pull out the messages which are the most important.
Ofter it really is only one in every thousand messages that might be relevant to the disaster response professionals.
The way that disasters are tipically responded to,  is that different organizations will take care of different parts of the problem. So, one organization will care about the watter, another will care about blocked roads, another will care about medical supplies, and so on

**Robert Munro - CTO, Figure Eight**

Figure Eight preprocess and pre-labels messages and tweets sent during real life disasters.
This projet consists on building an webbapp to classify messages into 36 categories, based on a supervised machine learning model, in order to help disaster response professionals to filter and respond the most relevant messages.



## Project Steps

*   **ETL Pipeline** - Extract data from the .csv files provided by Figure Eight, merge it into a PanDas dataframe, apply proper data cleaning and transformation and then save it into a sqlite3 database to be used at the supervised machine learning process;
*   **Machine Learning Pipeline** - Read data from the sqlite3 database, apply NLP (Natural Language Processing) techniques at the messages, and then run the machine running pipeline. Several models were tested, using the following metrics: Precision, Recall and F1-Score. Model hyper-parameters were optimized using GridSearchCV. Final model was saved into a pickle file;
*   **Webapp Development** - App built using Boostrap templates and Flask;
*   **Deploy** - Webapp was deployed into Heroku;

## Installation

Install the required libraries:

```sh
$ pip install -r requirements.txt
```

Running the ETL pipeline and generating the sqlite3 database (DisasterResponse.db)

```sh
$ python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

Running the machine learning pipeline

```sh
$ python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

Running the application

```sh
$ python disaster_response/run.py
```

## Libraries

The ETL pipeline, model training and application scrips uses the following libraries to run:

-   **Sys**
-   **Json**
-   **Plotly**
-   **Flask**
-   **PanDas**
-   **Re**
-   **Nltk** -  download ('punkt', 'stopwords', 'wordnet')
-  **Sklearn**
-  **Sqlalchemy**
-   **Numpy**
-   **Pickle**

## Files in the repository

The project is composed by the following files:

| File | Description |
| ------ | ------ |
| requirements.txt | list of required libraries |
| README.md | text file holding information about the project |
| Procfile | File that tells Heroku how to start the web app |
| models/classifier.pk | Trained model |
| models/tokenize_function.py | Data processing function |
| models/train_classifier.py | Script that runs the machile learning pipeline |
| disaster_response/run.py | Web app main script |
| disaster_response/template/master.html | Template of the web app's main page |
| disaster_response/template/go.html | Classification results template |
| data/disaster_categories.csv | Categories data used as train dataset |
| data/disaster_messages.csv | Messages used as train dataset |
| data/DisasterResponse.db | sqlite3 database generated by the ETL pipeline |
| data/process_data.py | Script that runs the ETL pipeline |

## Results

The best results for the training step were achieved using the following machile learning pipeline:

```sh
   pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tkf.tokenize, ngram_range = (1,2))),
        ('tfidf', TfidfTransformer()),
        ('clf',MultiOutputClassifier(OneVsRestClassifier(LinearSVC(), n_jobs=1)))  
                        ])
```

(Hyper-parameters where optimized using GridSearchCV)

where tkf.tokenize is:

```sh
def tokenize(text):
    
    #remove special characteres and swap numbers by 'digit'
    text = re.sub(r"[^a-zA-Z0-9]"," ",text)
    text = re.sub(r"[0-9]", "digit", text)

    #tokenize and remove stop words
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words("english")]

    # stemming
    stemmed = [SnowballStemmer('english').stem(t) for t in tokens]

    # instantianting the lemmatizer
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
```


Metrics per category:

| Category | Precision | Recall | F1-Score | Support |
| ------ | ------ | ------ | ------ | ------ |
| related | 0.85 | 0.93 | 0.89 | 3999 |
| request | 0.72 | 0.64 | 0.68 | 860 |
| offer | 0.00 | 0.00 | 0.00 | 26 |
| aid related | 0.70 | 0.77 | 0.73 | 2157 |
| medical help | 0.63 | 0.38 | 0.48 | 420 |
| medical products | 0.72 | 0.39 | 0.51 | 271 |
| search and rescue | 0.72 | 0.15 | 0.25 | 138 |
| secutiry | 0.33 | 0.01 | 0.02 | 92 |
| military | 0.62 | 0.36 | 0.46 | 177 |
| child alone | 0.00 | 0.00 | 0.00 | 0 |
| water | 0.71 | 0.71 | 0.71 | 320 |
| food | 0.79 | 0.77 | 0.78 | 573 |
| shelter | 0.74 | 0.62 | 0.68 | 465 |
| clothing | 0.80 | 0.45 | 0.58 | 91 |
| money | 0.50 | 0.23 | 0.31 | 111 |
| missing people | 0.54 | 0.12 | 0.20 | 57 |
| refugees | 0.64 | 0.24 | 0.35 | 177 |
| death | 0.74 | 0.46 | 0.57 | 240 |
| other aid | 0.51 | 0.25 | 0.34 | 691 |
| infrastructure related | 0.54 | 0.11 | 0.18 | 353 |
| transport | 0.63 | 0.23 | 0.34 | 232 |
| buildings | 0.69 | 0.36 | 0.47 | 295 |
| electricity | 0.54 | 0.26 | 0.35 | 98 |
| tools | 0.00 | 0.00 | 0.00 | 34 |
| hospitals | 0.62 | 0.07 | 0.13 | 67 |
| shops | 0.00 | 0.00 | 0.00 | 25 |
| aid centers | 1.00 | 0.02 | 0.03 | 63 |
| other infrastructure | 0.50 | 0.06 | 0.11 | 235 |
| weather related | 0.81 | 0.78 | 0.79 | 1489 |
| floods | 0.85 | 0.61 | 0.71 | 454 |
| storm | 0.71 | 0.66 | 0.68 | 503 |
| fire | 0.50 | 0.14 | 0.22 | 56 |
| earthquake | 0.90 | 0.76 | 0.82 | 528 |
| cold | 0.64 | 0.39 | 0.48 | 100 |
| other_waether | 0.56 | 0.19 | 0.28 | 284 |
| direct_report | 0.61 | 0.53 | 0.57 | 958 |

General model metrics:
| Metrics  | Precision | Recall | F1-Score | Support |
| ------ | ------ | ------ | ------ | ------ |
| Micro Average | 0.76 | 0.64 | 0.70 | 16639 |
| Macro Average | 0.59 | 0.35 | 0.41 | 16639 |
| Weighted Average | 0.73 | 0.64 | 0.66 | 16639 |
| Samples Average | 0.61 | 0.54 | 0.53 | 16639 |

As shown at the column "support" of the results table, the training dataset was extremely unbalances, which lead the model to perform badly at some categories and dragged down the macro results a bit.

Better results would be achieved by removing the categories which have no expressive representation or no representation at all, as we have on "child alone".

## Acknowledgments

* [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025)
*  [Figure Eight](https://www.figure-eight.com/)