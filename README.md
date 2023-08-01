# Back And Forth to Q&A
#
# For my project, I had to find a way to condense back and forth conversations on tech chat forums into Q&As.
# To extract the data, I built a wep scraper that was easily able to give me the 1000 data points I needed, while cleaning the data as well with regex expressions. Then, I preprocessed the data by grouping all the questions together and answers together for each conversation. However, I still needed train data, so I got 100 of the preprocessed data points and used ChatGPT to summarize them. For my model, I ended up using BART from the transformers library after failing to use smaller-scale models from spacy or nltk. Finally, I validated the results using the ROUGE scoring system and got an average score of 0.36 for questions and 0.26 for answers.
