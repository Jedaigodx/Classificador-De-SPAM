import pandas as pd
import re

df = pd.read_csv('spam.csv', encoding='latin-1', sep='\t', header=None, names=['label', 'message'])

print(df.head())

#função limpar limpar textos
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|https\S+','',text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# limpando dados inuteis
df['clean'] = df['message'].apply(clean_text)

print("\nExemplo de mensagem limpa:")

print(df[['message', 'clean']].head())
