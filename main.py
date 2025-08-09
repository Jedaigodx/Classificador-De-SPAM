import pandas as pd
import re

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
