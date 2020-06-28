

#importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import json
import matplotlib as rcParams
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from gensim.summarization.summarizer import summarize
from nltk.tokenize import word_tokenize, sent_tokenize
from wordcloud import WordCloud, STOPWORDS
import collections
from ludwig.api import LudwigModel
import math
import re
from collections import Counter
import pickle 
from flask import Flask, request, render_template
from flask import url_for
from flask import make_response
from ludwig.api import LudwigModel
import csv
import pandas as pd 

from collections import Counter
from jinja2 import Template

global result
app = Flask(__name__)
     
@app.route("/")
def hello():
    return render_template('runIndex.html')

@app.route('/save-comment', methods=['POST'])
def save_comment():
    if request.method == 'POST':
        
        name = request.form['text']
        
        with open('nameList.csv','w') as inFile:
          
            writer = csv.writer(inFile)
            writer.writerow(["abstract"])
            writer.writerow([name])


    #trained model        
    model_path = "results/experiment_run_5/model"
    model = LudwigModel.load(model_path)

    #loading dataset created during training
    test_df = pd.read_csv('/home/vineeta/datframe_with_two_clusters.csv',index_col=None, header=0,error_bad_lines=False)

    test_df = pd.DataFrame(test_df,columns = ['Index','cord_uid', 'sha', 'source_x', 'title', 'doi',
       'pmcid', 'pubmed_id', 'license', 'abstract', 'publish_time',
       'authors', 'journal', 'mag_id', 'who_covidence_id', 'arxiv_id',
       'pdf_json_files', 'pmc_json_files', 'url', 's2_id',
       'cluster_doc2vec_kmeans', 'cluster_doc2vec_kmeans4',
       'cluster_doc2vec_kmeans2'])


    #Using only 10000 rows for performing our CureCov.Lit query search, restricted due to system capabilities
    test_df_sliced=test_df[0:10000]



    #summarizing the abstracts of the paper in 1-2 lines
    test_df_sliced['summary']=""
    for (index_label, row_series) in test_df_sliced.iterrows():

        try:
            sentences = test_df_sliced.abstract[index_label].lower()
            s=summarize(sentences)
            #print('Row Index label : ', index_label)
            test_df_sliced['summary'][index_label] =  s
        except ValueError as ve:
            test_df_sliced['summary'][index_label] =  'NA'
            #print(ve)


    #top 5 keyword extraction of the titles to ease our search and give better recommendation of research papers 
    test_df_sliced['keywords']=""
    for (index_label, row_series) in test_df_sliced.iterrows():

        all_words_titles=(test_df_sliced.iloc[index_label]['title'])
        stopwords = STOPWORDS
        filtered_words = [word for word in all_words_titles.split() if word not in stopwords]
        counted_words = collections.Counter(filtered_words)

        words = []
        counts = []
        for letter, count in counted_words.most_common(5):
            words.append(letter)
            counts.append(count)
            
        keywords = ', '.join(str(x) for x in words)

        test_df_sliced['keywords'][index_label] =  keywords
        #print(words)
       


    test_df_sliced_predicted = test_df_sliced.assign(predicted_cluster=1) 

    #performing text classification on our original dataset
    model = LudwigModel.load("results/experiment_run_5/model")

    predictions = model.predict(test_df_sliced_predicted)
    print(predictions)
    test_df_sliced_predicted=test_df_sliced_predicted.join(predictions.cluster_doc2vec_kmeans2_predictions)[['Index','cord_uid', 'sha', 'source_x', 'title', 'doi', 'pmcid',
           'pubmed_id', 'license', 'abstract', 'publish_time', 'authors',
           'journal', 'mag_id', 'who_covidence_id', 'arxiv_id',
           'pdf_json_files', 'pmc_json_files', 'url', 's2_id',
           'cluster_doc2vec_kmeans', 'cluster_doc2vec_kmeans4',
           'cluster_doc2vec_kmeans2', 'summary', 'keywords', 'cluster_doc2vec_kmeans2_predictions']]


  


    WORD = re.compile(r"\w+")


    def get_cosine(vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
        sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator


    def text_to_vector(text):
        words = WORD.findall(text)
        return Counter(words)


    


    #creating cluster group specific dataframe for quick search
    test_df_clustergroups_sliced = test_df_sliced_predicted.groupby('cluster_doc2vec_kmeans2_predictions')

    test_df_sliced_predicted.groupby('cluster_doc2vec_kmeans2_predictions').groups

    test_df_group_0 = test_df_clustergroups_sliced.get_group('0')

    test_df_group_1 = test_df_clustergroups_sliced.get_group('1')

    test_df_group_0.info

    test_df_group_1.info



    #classifying our input query after saving it into a .csv to be loaded to ludwig  
    model_path = "results/experiment_run_5/model"
    model = LudwigModel.load(model_path)

    image = request.form["text"]

    test_y_predicted = pd.read_csv('/home/vineeta/flask/WebApp/nameList.csv',index_col=None, header=0,error_bad_lines=False)
    result = model.predict(test_y_predicted)
    data = test_y_predicted.abstract[0]
    score = test_y_predicted.join(predictions.cluster_doc2vec_kmeans2_predictions)[["abstract", "cluster_doc2vec_kmeans2_predictions"]]

    score = score.cluster_doc2vec_kmeans2_predictions[0]

    classified_dataset = 'test_df_group_'+score
 

    #calculating similarity between the abstracts of the input query and our dataset to find all the papers similar to our query
    similarity_list =[]
    for (index_label, row_series) in test_df_sliced.iterrows():
      
        if str(score) == '1':
            try:
                text2 = test_df_group_1.iloc[index_label]['abstract']
            except IndexError:
                print('indexerror')
        else:
            try:
                text2 = test_df_group_0.iloc[index_label]['abstract']
            except IndexError:
                print('indexerror')
        vector1 = text_to_vector(str(data))
        vector2 = text_to_vector(text2)

        cosine = get_cosine(vector1, vector2)
        similarity_list.append(cosine)
        print(index_label,"Cosine:", cosine)


    #finding index of papers with similar keywords as our input query to give better suggestions
    similar_list =[]
    x = test_df_sliced_predicted.keywords[similarity_list.index(max(similarity_list))]
    print(type(x))
    print(x)
    y=[]
    y=x.split(',')
    print(y)

    length = len(y) 
    print (length)

    for i in range(length):
     
        print(y[i])
        print(score)
        #print(test_df_group_0.keywordsx.str.contains(y[i])) 
        if score=='0':
            k = [j for j, c in enumerate(test_df_group_0.keywords.str.contains(y[i])) if c]
        else:
            k = [j for j, c in enumerate(test_df_group_1.keywords.str.contains(y[i])) if c]
     
        similar_list.append(k)
     

    flat_list = [item for sublist in similar_list for item in sublist]




    print(flat_list)


    temp = []

    for x in flat_list:
        if x not in temp:
            temp.append(x)

    flat_list = temp

    #output list of 10 most similar articles in the dataset with title, summary, direct link to the paper and published date. 
    output_list=[]
    print(len(flat_list))
    for i in range(len(flat_list)):
        output_list.append(test_df_sliced_predicted.iloc[i]['title'])
        output_list.append(test_df_sliced_predicted.iloc[i]['summary'])
        #doi_relevant = 'http://dx.doi.org/'+test_df_sliced.iloc[k]['doi']
        #print(doi_relevant)
        output_list.append('http://dx.doi.org/'+test_df_sliced_predicted.iloc[i]['doi']) 
        output_list.append('Published on: '+test_df_sliced_predicted.iloc[i]['publish_time'])
        print(test_df_sliced_predicted.iloc[i]['title'])
        print(test_df_sliced_predicted.iloc[i]['summary'])
        #doi_relevant = 'http://dx.doi.org/'+test_df_sliced.iloc[k]['doi']
        #print(doi_relevant)
        print('http://dx.doi.org/'+test_df_sliced_predicted.iloc[i]['doi']) 
        print('Published on: '+test_df_sliced_predicted.iloc[i]['publish_time'])
        print('')
        print('-------------------------------------------------------------------------------------------------')
        print('-------------------------------------------------------------------------------------------------')
     

    template = Template("""
    <table>
    {% for item, count in bye.items() %}
         <tr><td>{{item}}</td><td>{{count}}</td></tr>
    {% endfor %}
    </table>
    """)
    
    return template.render(bye=Counter(output_list[:41]))

if __name__ == "__main__":
    app.run(debug=True)



