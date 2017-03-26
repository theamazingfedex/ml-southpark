
##Using LSTM Cells in a recurrent neural network, this will generate chatbot profiles for each primary southpark character


```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
```


```python
df = pd.read_csv("../input/All-seasons.csv")
```


```python
lines = df["Line"]
characters = df["Character"]
episodes = df["Episode"]
charlines = "("+characters+") " + lines
text = ""
for line in charlines:
    text += line
```


```python
token_dict = { 
    '!': '||EXCLAIMATIONMARK||',
    '?': '||QUESTIONMARK||',
    '--': '||DOUBLEDASH||',
    '"': '||DOUBLEQUOTE||',
    ',': '||COMMA||',
    '.': '||PERIOD||',
    ';': '||SEMICOLON||',
    '\n': '||NEWLINE||',
    '(': '||OPENPAREN||',
    ')': '||CLOSEPAREN||',
    #'+': '||PLUS||',
    '&': '||AMPERSAND||',
    ':': '||COLON||',
    #'\'': '||APOSTROPHE||',
    #'-': '||DASH||',
}

for key,token in token_dict.items() :
    text = text.replace(key, ' {} '.format(token)) 
    
text = text.lower().split()
```


```python
vocab = set(text)
vocab_to_int = {w:i for i,w in enumerate(vocab)}
int_to_vocab = {i:w for w,i in vocab_to_int.items()}
int_text = [vocab_to_int[word] for word in text]
```

### Save preprocessed data to file


```python
pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))
```

## Checkpoint


```python
int_text, vocab_to_int, int_to_vocab, token_dict = pickle.load(open('preprocess.p', mode='rb'))
```
