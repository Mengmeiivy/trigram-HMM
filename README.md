# Trigram HMM

For both the English and Chinese part-of-speech tagging problem, I designed, implemented and tuned a Trigram HMM tagger. 

# English
For the English part, I used the Penn Treebank Wall Street Journal corpus. 
- WSJ_02-21.pos: the training file 
- WSJ_24.pos: the development file
- WSJ_23.words and WSJ_23.pos: the test files  

To run the program: python English_trigram.py

To evaluate the result: python score.py WSJ_23.pos english_output.txt. It should return an accuracy of 96.53%. 

# Chinese
For the Chinese part, I used the Penn Chinese Treebank. I preprocessed the data into the following tree parts:
- chinese_training.txt
- chinese_dev_pos.txt
- chinese_test_words.txt and chinese_test_pos.txt

To run the program: python Chinese_trigram.py

To evaluate the result: python score.py chinese_test_pos.txt chinese_output.txt. It should return an accuracy of 90.84%. 

*A final report on the project is also included here. 
