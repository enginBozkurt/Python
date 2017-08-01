from collections import Counter
import os
import pandas as pd
import matplotlib.pyplot as plt

text = "This is my string, string is a word for this string."



book_dir = "C:/Users/Lenovo/.spyder-py3/Books"

def count_words_fast(text):
    """
    Count the number of times each word occurs in the text (str). 
    Return dicitonary where keys are unique words and values are word counts.
    Skip punctuation.
    """
    word_counts = {}    #empty dictionary 
    text = text.lower()
    skips = [".", ":", ";", ",", "'", '"']
    for ch in skips:
        text = text.replace(ch, "")
        
    word_counts =  Counter(text.split(" "))
    return word_counts


def count_words(text):
    """
    Count the number of times each word occurs in the text (str). 
    Return dicitonary where keys are unique words and values are word counts.
    Skip punctuation.
    """
    word_counts = {}    #empty dictionary 
    text = text.lower()
    skips = [".", ":", ";", ",", "'", '"']
    for ch in skips:
            text = text.replace(ch, "")
        
       
    for word in text.split(" "):
       if word in word_counts:
           word_counts[word] += 1
       else:
           word_counts[word] = 1
    return word_counts

def read_book(title_path):
    """Read a book and return it as string"""
    with open(title_path, "r", encoding = "utf8") as current_file:
        text = current_file.read()
        text = text.replace("\n", "").replace("\r", "")
    return text

        
text = read_book("./Books/English/shakespeare/Romeo and Juliet.txt") 
ind = text.find("What's in a name?") 

def word_stats(word_counts):
    """ Return number of unique words and word frequencies"""
    num_unique = len(word_counts)
    counts = word_counts.values()
    return(num_unique, counts)

book_dir = "C:/Users/Lenovo/.spyder-py3/Books"

stats = pd.DataFrame(columns = ("language", "author", "title", "length", "unique"))
title_num = 1
for language in os.listdir(book_dir):
    for author in os.listdir(book_dir + "/" + language):
        for title in os.listdir(book_dir + "/" + language + "/" + author):
            inputfile = book_dir + "/" + language + "/" + author + "/" + title
            print(inputfile)
            text = read_book(inputfile)
            (num_unique, counts) = word_stats(count_words(text))
            stats.loc[title_num] = language, author.capitalize(), title.replace(".txt", ""), sum(counts), num_unique
            title_num += 1
            
            
            
            
#find a part of book and read from this index to desired index
text = read_book("C:/Users/Lenovo/.spyder-py3/Books/English/shakespeare/Romeo and Juliet.txt")
ind = text.find("What's in a name?")

sample_text = text[ind : ind +1000]

#find subdirectories
import os
book_dir = "C:/Users/Lenovo/.spyder-py3/Books"
os.listdir(book_dir)

#how to create a table using Panda DataFrame function
table = pd.DataFrame(columns = ("name","age"))
table.loc[1] = "John", 33
table.loc[2] = "James", 35

#plotting unique words and length
plt.figure(figsize = (10,10))
subset = stats[stats.language == "English"]
plt.loglog(subset.length, subset.unique, "o", label = "English", color = "crimson")

subset = stats[stats.language == "French"]
plt.loglog(subset.length, subset.unique, "o", label = "French", color = "forestgreen")

subset = stats[stats.language == "German"]
plt.loglog(subset.length, subset.unique, "o", label = "German", color = "orange")

subset = stats[stats.language == "Portuguese"]
plt.loglog(subset.length, subset.unique, "o", label = "Portuguese", color = "blueviolet")

plt.legend()
plt.xlabel("Book Length")
plt.ylabel("Number of unique words")
plt.savefig("lang_plot.pdf")
























         
         
         