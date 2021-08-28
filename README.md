# Arabic-Rating-ChatBot
Retrieval-based Arabic Rating ChatBot for my Smart Methods training. Using NLTK, Keras and TKinter 

## Requirements:
 - Spyder IDE (the arabic text is normal in anaconda applications, else you'll have to use the arabic reshaper)
 - Miniconda
 - keras 
 - nltk
 - python 3.7 (or less)


# steps:

```
conda create -n p37env python=3.7 (Create a virtual environment with this version or less)

conda activate p37env     

conda install keras 

conda install 'h5py==2.10.0' (downgrade h5py to avoid errors)

cd (to the folder you have your python file in)

python train_chatbot.py   
```
You'll get this:
```
# Fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1) 
#here we specified the number of epochs to be 200
```
![Screen Shot 2021-08-28 at 11 45 24 PM](https://user-images.githubusercontent.com/53378171/131231310-749b10b9-3d1b-439e-869d-7debfe66e555.png)

![Screen Shot 2021-08-28 at 11 45 55 PM](https://user-images.githubusercontent.com/53378171/131231312-b1374308-1cb1-417f-bbf3-376c8d27bb7b.png)

### Trying arabic packages: 
```
from nltk.stem.isri import ISRIStemmer #arabic stemmer
st = ISRIStemmer()
w= 'حركات'
m='يفعلون'
print("\t \t",st.stem(w),"   \n")
print("\t  \t",st.stem(m),"  \n")

isri_stemmer = ISRIStemmer()
stem_word = isri_stemmer.stem("فسميتموها")
print("\t\t\t\t\t\t\t",stem_word)
from nltk.corpus import stopwords
all_stopwords = stopwords.words('arabic')
print(all_stopwords)
```
![Screen Shot 2021-08-27 at 9 35 31 AM](https://user-images.githubusercontent.com/53378171/131231438-dd15f9b2-acf9-4d87-9daa-49a3c5f74210.png)

--- 

Now run the GUI for Pythin (Tinkter)
```
python GUI_ChatBot.py
```
# Final Result:
![Screen Shot 2021-08-29 at 12 10 11 AM](https://user-images.githubusercontent.com/53378171/131231185-9a3d0a6c-93fe-4ed8-ae68-08b9d6e75c3e.png)

## Problems faced:

- Unfortantley, when I ran the codes I made for cleaning Arabic text, the accuracy went way down, and the ChatBot responses didn't make any sense. So, I removed them and only kept a few lines so the accuracy would rise back again to 0.9
```
if w in Arabic_stop_words:
        words.remove(w)
    # Normalize alef variants to 'ا'
    nw = normalize_alef_ar(w)
    # Normalize alef maksura 'ى' to yeh 'ي'
    nw = normalize_alef_maksura_ar(w)
    # Normalize teh marbuta 'ة' to heh 'ه'
    nw = normalize_teh_marbuta_ar(w)
    # removing Arabic diacritical marks
    nw = dediac_ar(w)
    words.extend(nw)
```
- keras wasn't compatible with python3.8, and this caused lot of erros and thats why I had to create a virtual environemnt with lower version than I use: Python3.8
