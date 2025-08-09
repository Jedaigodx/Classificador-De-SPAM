import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

#importando o dataframe SPAM
df = pd.read_csv('spam.csv', encoding='latin-1')

df = df[['v1','v2']]
df.columns = ['label','message']

print("Dados originais:")
print(df.head())

#função limpar limpar textos
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# limpando dados inuteis
df['clean'] = df['message'].apply(clean_text)

print("\nExemplo de mensagem limpa:")
print(df[['message', 'clean']].head())

#transformando texto em números
vectorizer = TfidfVectorizer(stop_words='english',max_features=2000)
x = vectorizer.fit_transform(df['clean'])
y = df['label'].map({'ham':0,'spam':1})

print("\nFormato da matriz:", x.shape)

#dividindo treino e teste
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

#treinando o modelo 
model =  MultinomialNB()
model.fit(x_train,y_train)

#Avaliando o modelo Acurácia e precisão
y_pred = model.predict(x_test)

print("Acurácia",accuracy_score(y_test,y_pred))
print("\nrelatório de classificação:\n",classification_report(y_test,y_pred,target_names=['ham','spam']))
print("\nMatriz de confusão:\n",confusion_matrix(y_test,y_pred))