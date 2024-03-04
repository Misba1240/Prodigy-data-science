#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from tqdm.auto import tqdm 


# In[2]:


import nltk 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
import re 
from collections import Counter
from string import punctuation


# In[3]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import precision_score, recall_score , f1_score, accuracy_score


# In[4]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[5]:


from gensim.models import Word2Vec
import torch 
import torch.nn as nn  
from torch.optim import Adam
from torch.utils.data import DataLoader , TensorDataset
from torchmetrics import ConfusionMatrix 
from mlxtend.plotting import plot_confusion_matrix

lemma = WordNetLemmatizer()
lb = LabelEncoder()


# In[6]:


twitter = pd.read_csv("C:/Users/91992/Downloads/archive (2)/twitter_training.csv")


# In[7]:


twitter.head()


# In[8]:


twitter.tail()


# In[9]:


np.unique(twitter['Borderlands'])


# In[10]:


np.unique(twitter['Positive'])


# In[11]:


twitter = twitter.drop('2401',axis=1)


# In[12]:


twitter = twitter.rename(columns={"Borderlands": "F2", "im getting on borderlands and i will murder you all ,": "F1", "Positive": "Review"})


# In[13]:


twitter.head()


# In[14]:


twitter["tweets"] = twitter["F1"].astype(str) + " " + twitter["F2"].astype(str)
twitter = twitter.drop(["F1","F2"], axis=1)


# In[15]:


twitter_review = {key : value for value , key in enumerate(np.unique(twitter['Review']))}
twitter_review


# In[16]:


def getlabel(n) : 
    for x , y in twitter_Review.items() : 
        if y==n : 
            return x


# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt

# Check data types and unique values
print(twitter['Review'].dtype)
print(twitter['Review'].unique())

# Create a count plot
plt.figure(figsize=(8, 6))
sns.countplot(data=twitter, x='Review')
plt.title('Review Count')
plt.xlabel('Review')
plt.show()


# In[18]:


import seaborn as sns
import matplotlib.pyplot as plt

review_count = twitter['Review'].value_counts()
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

sns.set_theme(style='ticks', palette='muted')
color = sns.color_palette(palette='muted')
explode = [0.02] * len(review_count)

axes.pie(review_count.values, labels=review_count.index, autopct='%1.1f%%', colors=color, explode=explode)
axes.set_title('Percentage review')

plt.tight_layout()
plt.show()


# In[19]:


import nltk
nltk.download('stopwords')


# In[20]:


import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from string import punctuation
from collections import Counter

# Download the 'stopwords' resource
nltk.download('stopwords')

def Most_Words_used(tweets, num_of_words):
    all_text = ''.join(twitter[tweets].values)
    
    all_text = re.sub('<.*?>', '', all_text)  # HTML tags
    all_text = re.sub(r'\d+', '', all_text)  # numbers
    all_text = re.sub(r'[^\w\s]', '', all_text)  # special characters
    all_text = re.sub(r'http\S+', '', all_text)  # URLs or web links
    all_text = re.sub(r'@\S+', '', all_text)  # mentions
    all_text = re.sub(r'#\S+', '', all_text)  # hashtags
    
    words = all_text.split() 
    
    # remove puncs 
    punc = list(punctuation) 
    words = [word for word in words if word not in punc]
    
    # remove stopwords (now that 'stopwords' is downloaded)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if not word in stop_words]
    
    word_counts = Counter(words)
    top_words = word_counts.most_common(num_of_words)
    
    return top_words

top_words = Most_Words_used('tweets', 50)

xaxis = [word[0] for word in top_words]
yaxis = [word[1] for word in top_words]

plt.figure(figsize=(16, 5))
plt.bar(xaxis, yaxis)
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.title('Most Commonly Used Words', fontsize=25)
plt.xticks(rotation=65)
plt.subplots_adjust(bottom=0.15)
plt.show()


# In[21]:


import nltk
nltk.download('punkt')


# In[22]:


import nltk.data

print(nltk.data.path)


# In[23]:


import nltk
import pandas as pd

# Specify the data directory explicitly
nltk.data.path.append("F:\\Internships\\ProdigyInfotech")

# Download the 'punkt' and 'wordnet' resources if not already downloaded
nltk.download('punkt')
nltk.download('wordnet')

# Define your DataPrep function
def DataPrep(text):
    # Your existing code for text preprocessing here
    
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Remove punctuation, stopwords, and perform lemmatization (assuming 'lemma' is defined)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    words = [lemma.lemmatize(word) for word in words]

    # Join the cleaned words back into a text
    cleaned_text = ' '.join(words)
    
    return cleaned_text


# asssuming a dataframe "twitter" with column "tweets"
twitter['cleaned_tweets'] = twitter['tweets'].apply(DataPrep)


# In[24]:


print(f'There are around {int(twitter["cleaned_tweets"].duplicated().sum())} duplicated tweets, we will remove them.')


# In[25]:


twitter.drop_duplicates("cleaned_tweets", inplace=True)


# In[26]:


twitter['tweet_len'] = [len(text.split()) for text in twitter.cleaned_tweets]


# In[27]:


twitter = twitter[twitter['tweet_len'] < twitter['tweet_len'].quantile(0.995)]


# In[28]:


plt.figure(figsize=(16,5))
AX = sns.countplot(x='tweet_len', data=twitter[(twitter['tweet_len']<=1000) & (twitter['tweet_len']>10)], palette="Greens_r")
plt.title('Count of tweets with number of words', fontsize=30)
plt.yticks([])
AX.bar_label(AX.containers[0])
plt.ylabel('count')
plt.xlabel('')
plt.show()


# In[29]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(twitter['cleaned_tweets'], twitter['Review'], train_size=0.85, random_state=42)


# In[30]:


len(x_train) ,len(x_test)


# In[31]:


from sklearn.feature_extraction.text import TfidfVectorizer
vector = TfidfVectorizer()
vector.fit(x_train)
Feature_Names = vector.get_feature_names_out()
print("number of feature words: ", len(Feature_Names))


# In[32]:


from sklearn.feature_extraction.text import TfidfVectorizer
x_train = [text.lower() for text in x_train]
vector = TfidfVectorizer()
x_train = vector.fit_transform(x_train)  
x_test = [text.lower() for text in x_test]
x_test = vector.transform(x_test)


# In[33]:


y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)


# In[34]:


from sklearn.linear_model import LogisticRegression


LR = LogisticRegression(random_state=42)


LR.fit(x_train, y_train)


# In[35]:


Train_Accuracy = LR.score(x_train , y_train)


# In[36]:


LR_Predicted = LR.predict(x_test)

Test_Accuracy = accuracy_score(y_test , LR_Predicted) 

Test_Precision = precision_score(y_test , LR_Predicted , average='weighted')
Test_Recall = recall_score(y_test , LR_Predicted , average='weighted')
Test_F1score = f1_score(y_test , LR_Predicted , average='weighted')


# In[37]:


print(f"The training accuracy for logistic regression : {(Train_Accuracy*100):0.2f}%\n")
print(f"The testing accuracy for logistic regression : {(Test_Accuracy*100):0.2f}%\n")
print(f"The precision for logistic regression : {Test_Precision:0.2f}\n")
print(f"The recall for logistic regression : {Test_Recall:0.2f}\n")
print(f"The F1 score for logistic regression : {Test_F1score:0.2f}\n")


# In[38]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

LR_Confusion_Matrix = confusion_matrix(y_test , LR_Predicted)
sns.heatmap(LR_Confusion_Matrix, annot=True,fmt='4g')
plt.show()


# In[39]:


RF = RandomForestClassifier()
RF.fit(x_train , y_train)


# In[40]:


Train_Accuracy1 = RF.score(x_train , y_train)


# In[41]:


RF_Predicted = RF.predict(x_test)

Test_Accuracy1 = accuracy_score(y_test , RF_Predicted) 

Test_Precision1 = precision_score(y_test , RF_Predicted , average='weighted')
Test_Recall1 = recall_score(y_test , RF_Predicted , average='weighted')
Test_F1score1 = f1_score(y_test , RF_Predicted , average='weighted') 


# In[42]:


print(f"The training accuracy for Random Forest : {(Train_Accuracy1*100):0.2f}%\n")
print(f"The testing accuracy for Random Forest : {(Test_Accuracy1*100):0.2f}%\n")
print(f"The precision for Random Forest : {Test_Precision1:0.2f}\n")
print(f"The recall for Random Forest : {Test_Recall1:0.2f}\n")
print(f"The f1 score for Random Forest : {Test_F1score1:0.2f}\n")


# In[43]:


RF_CM = confusion_matrix(y_test , RF_Predicted)
sns.heatmap(LR_Confusion_Matrix, annot=True,fmt='3g')
plt.show()


# In[44]:


Max_Length = np.max(twitter['tweet_len'])
Max_Length


# In[45]:


def lstm_prep(column , seq_len) : 
    corpus = [word for text in column for word in text.split()]
    words_count = Counter(corpus) 
    sorted_words = words_count.most_common()
    vocab_to_int = {w:i+1 for i , (w,c) in enumerate(sorted_words)}
    
    text_int = [] 
    
    for text in column : 
        token = [vocab_to_int[word] for word in text.split()]
        text_int.append(token)
        features = np.zeros((len(text_int) , seq_len) , dtype = int)
    for idx , y in tqdm(enumerate(text_int)) : 
        if len(y) <= seq_len : 
            zeros = list(np.zeros(seq_len - len(y)))
            new = zeros + y
            
        else : 
            new = y[:seq_len]
            
        features[idx,:] = np.array(new)
        
    return sorted_words, features


# In[46]:


VOCAB , tokenized_column = lstm_prep(twitter['cleaned_tweets'] , Max_Length)


# In[47]:


VOCAB[:10]


# In[48]:


len(VOCAB)


# In[49]:


tokenized_column.shape


# In[50]:


def most_common_words(vocab):
    keys = []
    values = []
    
    # Unpack the keys and values from the vocab list of tuples
    for key, value in vocab[:30]:
        keys.append(key)
        values.append(value)
        
    plt.figure(figsize=(15, 5))
    ax = sns.barplot(x=keys, y=values, palette='mako')
    plt.title('Top 20 most common words', size=25)
    ax.bar_label(ax.containers[0])
    plt.ylabel("Words count")
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.15)
    plt.show()

most_common_words(VOCAB)


# In[51]:


X = tokenized_column
y = lb.fit_transform(twitter['Review'].values)


# In[52]:


X_train , X_test , Y_train , Y_test = train_test_split(X , y , train_size=0.85 , random_state=42)


# In[53]:


train_data = TensorDataset(torch.from_numpy(X_train), torch.LongTensor(Y_train))
test_data = TensorDataset(torch.from_numpy(X_test), torch.LongTensor(Y_test))


# In[54]:


BATCH_SIZE = 64


# In[55]:


torch.manual_seed(42)
train_dataloader = DataLoader(
    dataset = train_data , 
    batch_size=BATCH_SIZE , 
    shuffle=True
)


# In[56]:


torch.manual_seed(42) 
val_dataloader = DataLoader(
    dataset = test_data , 
    batch_size = BATCH_SIZE , 
    shuffle=False
)


# In[57]:


print(f"the size of the train dataloader {len(train_dataloader)} batches of {BATCH_SIZE}")


# In[58]:


print(f"the size of the test dataloader {len(val_dataloader)} batches of {BATCH_SIZE}")


# In[59]:


EMBEDDING_DIM = 200


# In[60]:


Word2vec_train_data = list(map(lambda x: x.split(), twitter['cleaned_tweets']))
word2vec_model = Word2Vec(Word2vec_train_data, vector_size=EMBEDDING_DIM)


# In[61]:


def weight_matrix(model,vocab):
    vocab_size= len(vocab)+1
    embedding_matrix = np.zeros((vocab_size,EMBEDDING_DIM))
    for word, token in vocab:
        if model.wv.__contains__(word):
            embedding_matrix[token]=model.wv.__getitem__(word)
    return embedding_matrix


# In[62]:


embedding_vec = weight_matrix(word2vec_model,VOCAB)
print("Embedding Matrix Shape:", embedding_vec.shape)


# In[63]:


def param_count(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    print('The Total number of parameters in the model : ', sum(params))


# In[64]:


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, hidden_dim, out_channels, bidirectional):
        super().__init__()

        self.no_layers = num_layers
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.num_directions = 2 if bidirectional else 1
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout=0.5,
            bidirectional=bidirectional,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim * self.num_directions, out_channels)

    def forward(self, x):
        h0 = torch.zeros((self.no_layers * self.num_directions, x.size(0), self.hidden_dim))
        c0 = torch.zeros((self.no_layers * self.num_directions, x.size(0), self.hidden_dim))

        embedded = self.embedding(x)

        out, _ = self.lstm(embedded, (h0, c0))

        out = out[:, -1, :]

        out = self.fc(out)

        return out


# In[65]:


VOCAB_SIZE = len(VOCAB) + 1
NUM_LAYERS = 2 
OUT_CHANNELS = 4 
HIDDEN_DIM = 256
BIDIRECTIONAL = True

model = LSTM(VOCAB_SIZE , EMBEDDING_DIM , NUM_LAYERS , HIDDEN_DIM , OUT_CHANNELS , BIDIRECTIONAL)

model.embedding.weight.data.copy_(torch.from_numpy(embedding_vec))

model.embedding.weight.requires_grad = True


# In[66]:


param_count(model)


# In[67]:


criterion=nn.CrossEntropyLoss()
optimizer=Adam(model.parameters(),lr=0.001)


# In[68]:


epochs = 10 
training_loss = []
training_acc = [] 
for i in tqdm(range(epochs)) : 
    epoch_loss = 0
    epoch_acc = 0 
    for batch , (x_train , y_train) in enumerate(train_dataloader) : 
        y_pred = model(x_train)
        
        loss = criterion(y_pred , y_train) 
        
        if batch % 500 == 0:
            print(f"Looked at {batch * len(x_train)}/{len(train_dataloader.dataset)} samples.")
            
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        epoch_loss +=loss 
        epoch_acc += accuracy_score(y_train , y_pred.argmax(dim=1))
        
    training_loss.append((epoch_loss/len(train_dataloader)).detach().numpy())
    training_acc.append(epoch_acc/len(train_dataloader))
    
    print(f"Epoch {i+1}: Accuracy: {(epoch_acc/len(train_dataloader)) * 100}, Loss: {(epoch_loss/len(train_dataloader))}\n\n")


# In[69]:


print(f"The loss of the training set is : {training_loss[-1]:0.2f}")
print(f"The accuracy of the training set is : {(training_acc[-1]*100):0.2f}%")


# In[70]:


plt.plot(range(epochs),training_loss,color="blue",label="Loss")
plt.plot(range(epochs),training_acc,color="green",label="Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy / Loss')
plt.legend()
plt.show()


# In[71]:


val_loss=0
val_acc3= 0
lstm_preds=[]
val_targets = []
torch.manual_seed(42)
with torch.no_grad() : 
        for x_val , y_val in tqdm(val_dataloader) : 
            y_pred=model.forward(x_val)
            val_pred = torch.softmax(y_pred , dim=1 ).argmax(dim=1)
            lstm_preds.append(val_pred)
            val_targets.extend(y_val)
            
            loss=criterion(y_pred,y_val)
            val_loss+=loss
            val_acc3 += accuracy_score(y_val , y_pred.argmax(dim=1))
            
            
val_loss/=len(val_dataloader)
val_acc3/=len(val_dataloader)
lstm_preds = torch.cat(lstm_preds)
val_targets = torch.Tensor(val_targets)


# In[72]:


train_acc3 = training_acc[-1]
val_precision3 = precision_score(val_targets,lstm_preds,average='weighted')
val_recall3 = recall_score(val_targets,lstm_preds,average='weighted')
val_f1score3 = f1_score(val_targets,lstm_preds,average='weighted')


# In[73]:


print(f"The training accuracy for LSTM : {(train_acc3*100):0.2f}%\n")
print(f"The validation accuracy for LSTM : {(val_acc3*100):0.2f}%\n")
print(f"The precision for LSTM : {val_precision3:0.2f}\n")
print(f"The recall for LSTM : {val_recall3:0.2f}\n")
print(f"The f1 score for LSTM : {val_f1score3:0.2f}\n")
print(f"The training loss for LSTM : {training_loss[-1]:0.2f}\n")
print(f"The validation loss for LSTM : {val_loss:0.2f}\n")


# In[74]:


confmat = ConfusionMatrix(num_classes=4, task='multiclass')
confmat_tensor = confmat(preds=lstm_preds,
                         target=val_targets)

fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),
    class_names=twitter_review.keys(),
    figsize=(10, 7)
)


# In[75]:


train_scores=[Train_Accuracy,Train_Accuracy1,train_acc3]
val_scores=[Test_Accuracy,Test_Accuracy1,val_acc3]

models = ['Logistic Regression','RandomForest','LSTM']

x = np.arange(len(models))

width = 0.25

fig, ax = plt.subplots(figsize=(20, 10))

rects1 = ax.bar(x - width, train_scores, width, label='Train Accuracy')

rects2 = ax.bar(x + width, val_scores, width, label='Validation Accuracy')

ax.set_xlabel('Models')
ax.set_ylabel('Accuracy')
ax.set_title('Comparison of Training and Validation Accuracies')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.3f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()


# In[76]:


test_twitter = pd.read_csv("C:/Users/91992/Downloads/archive (2)/twitter_validation.csv")


# In[77]:


test_twitter.columns


# In[78]:


test_twitter = test_twitter.drop('3364' , axis=1)


# In[79]:


test_twitter = test_twitter.rename(columns={"Facebook":"Feature2","I mentioned on Facebook that I was struggling for motivation to go for a run the other day, which has been translated by Tom’s great auntie as ‘Hayley can’t get out of bed’ and told to his grandma, who now thinks I’m a lazy, terrible person 🤣":"Feature1","Irrelevant": "labels"})


# In[80]:


test_twitter["tweets"]= test_twitter["Feature1"].astype(str) +" "+ test_twitter["Feature2"].astype(str)
test_twiter = test_twitter.drop(["Feature1","Feature2"],axis=1)


# In[81]:


test_twitter.head()


# In[82]:


twitter_Review = {
    'Positive': 0,
    'Neutral': 1,
    'Negative': 2,
   
}


# In[83]:


def getlabel(n):
    for label, code in twitter_Review.items():
        if code == n:
            return label


# In[84]:


label = getlabel(1)


# In[85]:


import torch
from tqdm import tqdm

def make_predictions(row):
    random_data = row.sample(n=10)
    random_tweets = random_data['tweets'].values

    cleaned_tweets = [] 
    for tweet in random_tweets:
        cleaned_tweets.append(DataPrep(tweet))
        
    x_test = vector.transform(cleaned_tweets).toarray()
    y_test = random_data['labels'].values

    lr_pred = LR.predict(x_test)
    rf_pred = RF.predict(x_test)

    _, X_test = lstm_prep(cleaned_tweets, Max_Length)
    X_test = torch.from_numpy(X_test)

    lstm_pred = model(X_test)
    lstm_pred = torch.softmax(lstm_pred, dim=1).argmax(dim=1)

    for i in tqdm(range(10)):
        print(f"The original tweet: {random_tweets[i]}\n")
        print(f"The original label: {y_test[i]}\n")
        print(f"The lr prediction is: {getlabel(lr_pred[i])}\n")
        print(f"The rf prediction is: {getlabel(rf_pred[i])}\n")
        print(f"The lstm prediction is: {getlabel(lstm_pred[i])}\n")
        print('-' * 120)

# Call the function with your data
make_predictions(test_twitter)

