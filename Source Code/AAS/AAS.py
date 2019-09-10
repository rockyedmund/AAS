from flask import Flask, render_template, url_for, flash, request, redirect
from werkzeug.utils import secure_filename
import os, re, pickle, csv
import numpy as np
from random import randint

app = Flask(__name__)


#Preprocessing
#Clean NER
def cleanNER(essay) :

    if re.search('@NUM\d', essay) or re.search('@TIME\d', essay) :
        essay=re.sub('@NUM\d',"123",essay)
        if re.search('@\w*', essay) :
            essay=re.sub('@\w*',"Sunday",essay)

    elif re.search('@\w*', essay) :
            essay=re.sub('@\w*',"Sunday",essay)
    
    return(essay)

#tokenization
from nltk.tokenize import sent_tokenize, TweetTokenizer
TT=TweetTokenizer()
def wordTokenization(essay) :
    return TT.tokenize(essay)

def sentTokenization(essay) :
    return sent_tokenize(essay)

#remove punctuation
import string
def removePunct(word_token) :
    punctuation=set(string.punctuation)
    checker=0
    temp_word_token=[]
    for i in range(len(word_token)): 
        for x in punctuation:
            if x == word_token[i]:
                checker = 1
                break;
        if checker == 0:
            temp_word_token.append(word_token[i])
        checker=0
    return temp_word_token

def wordCount(word) :
    return len(word)

def sentCount(sent) :
    return len(sent)

def averageWordLength(word, wordcount) :
    total = 0
    for i in word :
        total += len(i)
        
    return(total/wordcount)

def longWordCount(word) :
    longcheck = []
    for i in word :
        if len(i) > 7 :
            longcheck.append(i)
    return len(longcheck)

def averageSentLength(sent,sentcount) :
    total = 0
    for i in sent :
        a = removePunct(TT.tokenize(i))
        total += len(a)
    
    return(total/sentcount)

def longSentCount(sent) :
    longcheck = []
    for i in sent :
        a = removePunct(TT.tokenize(i))
        if len(a) > 15 :
            longcheck.append(a)
    return len(longcheck)

from nltk.stem import PorterStemmer
def wordStem(word) :
    ps = PorterStemmer()
    temp_word_token=[]
    for i in word:
        if ps.stem(i)[-1:] == "'" :
            temp_word_token.append(ps.stem(ps.stem(i)[:-1]))
        elif re.search("'\w*", i) :
            temp_word_token.append(ps.stem(re.sub("'\w*","",i) ))
        else :
            temp_word_token.append(ps.stem(i))
    return temp_word_token

def uniqueWordCount(word) :
    return len(set(word))

import nltk
def posTag(word) :
    return nltk.pos_tag(word)

def nounCount(word) :
    temp_word=[]
    for i in word:
        if i[1] == 'NN' or i[1] == 'NNS':
            temp_word.append(i)
    return len(temp_word)

def propernounCount(word) :
    temp_word=[]
    for i in word:
        if i[1] == 'NNP' or i[1] == 'NNPS':
            temp_word.append(i)
    return len(temp_word)

def adjCount(word) :
    temp_word=[]
    for i in word:
        if i[1] == 'JJ' or i[1] == 'JJR' or i[1] == 'JJS' :
            temp_word.append(i)
    return len(temp_word)

def verbCount(word) :
    temp_word=[]
    for i in word:
        if i[1] == 'VB' or i[1] == 'VBD' or i[1] == 'VBG' or i[1] == 'VBN' or i[1] == 'VBP' or i[1] == 'VBZ'  :
            temp_word.append(i)
    return len(temp_word)

def adverbCount(word) :
    temp_word=[]
    for i in word:
        if i[1] == 'RB' or i[1] == 'RBR' or i[1] == 'RBS' :
            temp_word.append(i)
    return len(temp_word)

def tenseRatio(word,verb_count) :
    present = []
    past = []
    temp_word=[]
    
    for i in word:
        if i[1] == 'VB' or i[1] == 'VBZ' or i[1] == 'VBG' or i[1] == 'VBP' :
            present.append(i)
        elif i[1] == 'VBD' or i[1] == 'VBN':
            past.append(i)
    
    
    if len(present) > len(past) :
        return(len(present)/verb_count)
    elif len(past) > len(present):
        return(len(past)/verb_count)
    else :
        return(0)

import language_check
def errorCount(essay) :
    tool = language_check.LanguageTool('en-US')
    checker = []
    matches = tool.check(essay)
    for i in matches :
        checker.append(i.fromx)
    return(len(set(checker)))


#ML Model
def txtClassifier(content, classifier, sc):
    
    #Clean NER
    content = cleanNER(content)

    #Tokenize
    content_wordtoken = wordTokenization(content)
    content_senttoken = sentTokenization(content)

    #remove punctuation
    content_wordtokennopunct = removePunct(content_wordtoken)

    #feature extraction
    temp_feature = []

    temp_feature.append(wordCount(content_wordtokennopunct))
    temp_feature.append(sentCount(content_senttoken))
    temp_feature.append(averageWordLength(content_wordtokennopunct, temp_feature[0]))
    temp_feature.append(longWordCount(content_wordtokennopunct))
    temp_feature.append(averageSentLength(content_senttoken,temp_feature[1]))
    temp_feature.append(longSentCount(content_senttoken))

    content_tokenstemmed = wordStem(content_wordtokennopunct)
    temp_feature.append(uniqueWordCount(content_tokenstemmed))

    content_tagged = posTag(content_wordtokennopunct)
    temp_feature.append(nounCount(content_tagged))
    temp_feature.append(propernounCount(content_tagged))
    temp_feature.append(adjCount(content_tagged))
    temp_feature.append(verbCount(content_tagged))
    temp_feature.append(adverbCount(content_tagged))

    temp_feature.append(tenseRatio(content_tagged,temp_feature[10]))
    temp_feature.append(randint(0, 30))

    scaled_feature = sc([temp_feature])
    
    ml_result = classifier.predict(scaled_feature)

    if ml_result[0] == 1 :
        ml_result = 1
    elif ml_result[0] == 2 :
        ml_result = 2
    else :
        ml_result = 3
    # ml_result = float(ml_result[0]) 
    # temp_feature = temp_feature.tolist()
    temp_feature[2] = round(temp_feature[2])
    temp_feature[4] = round(temp_feature[4])
    temp_feature[12] = round(temp_feature[12],2)
    output = temp_feature + [ml_result]

    return(output)

def csvClassifier(essay, classifier, sc) :

    for i in range(len(essay)) :
        essay[i]=cleanNER(essay[i])

    word_token = []
    sent_token=[]
    for i in essay:
        word_token.append(wordTokenization(i))
        sent_token.append(sentTokenization(i))

    word_token_nopunct=[]
    for i in word_token :
        word_token_nopunct.append(removePunct(i))
    
    word_count = []
    for i in word_token_nopunct :
        word_count.append(wordCount(i))

    sent_count = []
    for i in sent_token :
        sent_count.append(sentCount(i))

    average_word_length = []
    for i in range(len(word_count)) :
        average_word_length.append(averageWordLength(word_token_nopunct[i],word_count[i]))

    longword_count = []
    for i in word_token_nopunct :
        longword_count.append(longWordCount(i))

    average_sent_length = []
    for i in range(len(sent_count)) :
        average_sent_length.append(averageSentLength(sent_token[i],sent_count[i]))

    longsent_count = []
    for i in sent_token :
        longsent_count.append(longSentCount(i))

    word_token_stemmed = []
    for i in word_token_nopunct :
        word_token_stemmed.append(wordStem(i))
    unique_word_count = []
    for i in word_token_stemmed:
        unique_word_count.append(uniqueWordCount(i))

    tagged = []
    for i in word_token_nopunct:
        tagged.append(posTag(i))
    noun_count=[]
    for i in tagged :
        noun_count.append(nounCount(i))
    propernoun_count=[]
    for i in tagged :
        propernoun_count.append(propernounCount(i))
    adj_count=[]
    for i in tagged :
        adj_count.append(adjCount(i))
    verb_count=[]
    for i in tagged :
        verb_count.append(verbCount(i))
    adverb_count=[]
    for i in tagged :
        adverb_count.append(adverbCount(i))

    tense_ratio = []
    for i in range(len(tagged)) :
        tense_ratio.append(tenseRatio(tagged[i],verb_count[i]))

    error_count = []
    for i in essay :
        error_count.append(randint(0, 30))

    temp_feature= list(map(list, zip(*[word_count, sent_count, average_word_length, longword_count
                    , average_sent_length, longsent_count,  unique_word_count, noun_count, propernoun_count, adj_count
                    , verb_count, adverb_count, tense_ratio, error_count])))
    

    scaled_feature = sc(np.array(temp_feature))
    
    ml_result = classifier.predict(scaled_feature)
    ml_result = ml_result.tolist()

    output = []
    for i in range(len(essay)) :
        temp_feature[i][2] = round(temp_feature[i][2])
        temp_feature[i][4] = round(temp_feature[i][4])
        temp_feature[i][12] = round(temp_feature[i][12],2)
        output.append(temp_feature[i]+[ml_result[i]])

    return(output)

def nonmlScoring (cutter, ft):
    temp=[]
    for i in range(len(ft)):

        if i in (5,13) :
            if ft[i] > cutter[i][0] :
                temp.append(1)
            elif ft[i] <= cutter[i][0] and ft[i] > cutter[i][0] :
                temp.append(2)
            else:
                temp.append(3)

        else:

            if ft[i] < cutter[i][0] :
                temp.append(1)
            elif ft[i] >= cutter[i][0] and ft[i] < cutter[i][1] :
                temp.append(2)
            else:
                temp.append(3)

    y_pred = (round(sum(temp)/14))
        
    return(y_pred)    

def combine1(ml, nonml, w):
    y_c1 = []
    for i in range(len(ml)):
        y_c1.append(round((ml[i]*w[0]+nonml[i]*w[1])/sum(w)))
    
    return(y_c1)

def translateToClass(y):
    if y == 1 :
        return("Class 1: Bad")
    elif y == 2 :
        return("Class 2: Medium")
    else :
        return("Class 3: Good")

def countClass(output):
    class_count = [0,0,0]
    for i in output :
        if i == 1 :
            class_count[0] +=1
        elif i == 2:
            class_count[1] +=1
        else :
            class_count[2] +=1
    return(class_count)
    

#Upload file
UPLOAD_FOLDER = 'D:/Assignment/FYP2_1141128343/Source Code/AAS/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'secret_key'

def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/",methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
def home():
    header_csv = ["essay_id", "essay", "word_count", "sent_count", "ave_word_length", "long_word_count"
              , "ave_sent_length", "long_sent_count",  "unique_word_count", "noun_count", "propernoun_count", "adj_count"
              , "verb_count", "adverb_count", "tense_ratio", "error_count", "ml_y", "nonml_y", "c1_y", "c2_y"]
    output= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    filetype = 0
    with open('classifierml.pkl', 'rb') as file: 
        classifierml = pickle.load(file)
    with open('scalerml.sav', 'rb') as file: 
        scalerml = pickle.load(file)
    with open('classifierc2.pkl', 'rb') as file: 
        classifierc2 = pickle.load(file)
    with open('scalerc2.sav', 'rb') as file: 
        scalerc2 = pickle.load(file)

    mean_cut = [[233.19, 405.07], [13.13, 25.28], [4.08, 4.24], [18.69, 38.13], [22.58, 17.53], 
                [6.31, 11.15], [102.2, 167.97], [44.59, 79.21], [7.96, 17.07], [15.04, 27.24], [49.42, 83.68], 
                [13.02, 23.37], [0.8, 0.76], [12.93, 16.06]]
    median_cut = [[212.75, 391.0], [12.0, 24.5], [4.11, 4.27], [17.0, 38.5], [17.71, 16.2], [5.5, 10.5], 
                [97.0, 165.5], [41.5, 81.0], [5.0, 13.5], [13.5, 26.5], [44.5, 78.5], [11.0, 21.0], 
                [0.85, 0.81], [10.0, 14.0]]
    ratio_cut = [[83.31, 886.53], [8.11, 80.17], [3.24, 4.96], [11.76, 130.84], [247.15, 24.81], [29.17, 2.62], 
                [37.33, 374.85], [20.3, 225.84], [24.42, 271.68], [6.07, 67.5], [20.47, 217.67], [6.52, 72.5], 
                [0.07, 0.83], [127.5, 11.46]]

    weight = [1.0,1.0]

    in_ft = []
    # in_weight = []
    
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            print("-----------------------------no file1-----------------------------")
            # return redirect(request.url)
            return render_template('home.html', output=output, filetype=filetype, ratio_cut=mean_cut)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            print("-----------------------------no file2-----------------------------")
            return render_template('home.html', output=output, filetype=filetype, ratio_cut=mean_cut)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("-----------------File Inserted------------------")

            in_ft.append([request.form.get('inft0min'),request.form.get('inft0max')])
            in_ft.append([request.form.get('inft1min'),request.form.get('inft1max')])
            in_ft.append([request.form.get('inft2min'),request.form.get('inft2max')])
            in_ft.append([request.form.get('inft3min'),request.form.get('inft3max')])
            in_ft.append([request.form.get('inft4min'),request.form.get('inft4max')])
            in_ft.append([request.form.get('inft5min'),request.form.get('inft5max')])
            in_ft.append([request.form.get('inft6min'),request.form.get('inft6max')])
            in_ft.append([request.form.get('inft7min'),request.form.get('inft7max')])
            in_ft.append([request.form.get('inft8min'),request.form.get('inft8max')])
            in_ft.append([request.form.get('inft9min'),request.form.get('inft9max')])
            in_ft.append([request.form.get('inft10min'),request.form.get('inft10max')])
            in_ft.append([request.form.get('inft11min'),request.form.get('inft11max')])
            in_ft.append([request.form.get('inft12min'),request.form.get('inft12max')])
            in_ft.append([request.form.get('inft13min'),request.form.get('inft13max')])

            for i in in_ft:
                for j in i:
                    if j == '':
                        continue
                    try:
                        float(j)
                    except:
                        return render_template('home.html', output=output, filetype=filetype, ratio_cut=mean_cut)

            cutter = []
            for i in range(len(in_ft)) :
                temp=[]
                if in_ft[i][0] == '':
                    temp.append(mean_cut[i][0])
                else:
                    temp.append(float(in_ft[i][0]))

                if in_ft[i][1] == '':
                    temp.append(mean_cut[i][1])
                else:
                    temp.append(float(in_ft[i][1]))

                cutter.append(temp)

            # in_weight.append(request.form.get('inweight1'))
            # in_weight.append(request.form.get('inweight2'))

            # if (in_weight[0] == '' and in_weight[1] == '') :
            #     in_weight=weight
            # elif (in_weight[0] == '' and in_weight[1] != '') or  (in_weight[0] != '' and in_weight[1] == ''):
            #     return render_template('home.html', output=output, filetype=filetype, ratio_cut=mean_cut)

            # else:
            #     try:
            #         float(in_weight[0])
            #         float(in_weight[1])
            #         weight=in_weight
            #     except:
            #         print("ASfgggggggggggggg")
            #         return render_template('home.html', output=output, filetype=filetype, ratio_cut=mean_cut)


            if re.search(".*.txt", filename) :
                print("Text file inserted!")
                textfile = open("uploads/"+filename,'r')
                content = textfile.read()
                textfile.close()
                output = txtClassifier(content, classifierml, scalerml)
                ml_y = output[14]
                print(output)
                del output[-1]
                nonml_y = nonmlScoring(cutter,output)

                
                #combine1
                c1_y = combine1([ml_y],[nonml_y],weight)[0]

                #combine2
                X_c2 = output
                X_c2.append(nonml_y)
                    
                X_c2 = scalerc2([X_c2])

                c2_y = classifierc2.predict(np.array(X_c2))[0]

                #
                all_y=[translateToClass(ml_y),translateToClass(nonml_y),translateToClass(c1_y),translateToClass(c2_y)]

                output_csv = [1,content]
                for i in output:
                    output_csv.append(i)
                
                output_csv.append(ml_y)
                output_csv.append(nonml_y)
                output_csv.append(c1_y)
                output_csv.append(c2_y)
    

                with open("results/result.csv", 'w', newline='') as myfile:
                    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                    wr.writerow(header_csv)
                    wr.writerow(output_csv)

                filetype = 1

                return render_template('home.html', filetype=filetype, ratio_cut=mean_cut, output=output, all_y=all_y)

            elif re.search(".*.csv", filename) :
                print("Csv file inserted!")
                with open("uploads/"+filename, encoding="utf8", errors='ignore') as csvfile :
                    reader = csv.reader(csvfile, delimiter=',')
                    next(reader) #skip header
                    data = []
                    for row in reader:
                        data.append(row)

                essay_id = []
                essay = []
                for i in range(len(data)):
                    essay_id.append(data[i][0])
                    essay.append(data[i][1])

                output = csvClassifier(essay, classifierml, scalerml)


                output_csv = []
                for i in range(len(essay)) :
                    output_csv.append([essay_id[i]] + [essay[i]] + output[i])

                ml_y = []

                for i in output:
                    ml_y.append(int(i[14]))

                ml_class_count = countClass(ml_y)

                f = list(map(list, zip(*output)))
                del f[-1]

                averagef = []
                for i in range(len(f)) :
                    if i == 12:
                        averagef.append(round(sum(f[i])/len(f[i]),2))
                    else :
                        averagef.append(round(sum(f[i])/len(f[i])))

                minf = [min(i) for i in f]
                maxf = [max(i) for i in f]

                #non ml
                nonml_y = []

                for i in range(len(output)):
                    del output[i][14]

                for i in range(len(output)):
                    nonml_y.append(int(nonmlScoring(cutter,output[i])))

                nonml_class_count = countClass(nonml_y)

                #combine1
                c1_y = combine1(ml_y,nonml_y,weight)

                c1_class_count = countClass(c1_y)

                #combine2
                X_c2 = output

                for i in range(len(X_c2)):
                    X_c2[i].append(nonml_y[i])


                X_c2 = scalerc2(X_c2)
                c2_y = classifierc2.predict(X_c2)

                c2_class_count = countClass(c2_y)

                all_class_count = [ml_class_count, nonml_class_count, c1_class_count, c2_class_count]

                output_csv = []
                for i in range(len(ml_y)) :
                    temp = []
                    temp.append(essay_id[i])
                    temp.append(essay[i])
                    for j in output[i]:
                        temp.append(j)
                    temp.append(ml_y[i])
                    temp.append(nonml_y[i])
                    temp.append(c1_y[i])
                    temp.append(c2_y[i])
                    output_csv.append(temp)

                with open("results/result.csv", 'w', newline='') as myfile:
                    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                    wr.writerow(header_csv)
                    for i in output_csv:
                        wr.writerow(i)

                filetype = 2

                return render_template('home.html', output=output, filetype=filetype, all_class_count=all_class_count, averagef=averagef, 
                    minf=minf, maxf=maxf, ratio_cut=mean_cut)

            

    return render_template('home.html', output=output, filetype=filetype, ratio_cut=mean_cut)
    # return render_template('home.html', posts=posts)
from flask import send_file

@app.route('/return-files/')
def return_files_tut():
    try:
        return send_file('./results/result.csv', attachment_filename='result.csv', as_attachment = True)
    except Exception as e:
        return str(e)
    



if __name__ == '__main__':
    app.run(debug=True)
