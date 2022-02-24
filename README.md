
# Deceptive Online Review Detection

Since I arrived in the United States 6 months ago, I had to do a lot of online shopping and rely on many online reviews. However, reviews for many products were contradicting each other. So, I was curious whether some of these reviews are fake. Upon more research, I found that online shopping platforms conceded that there actually are significant amount of fake reviews out there. So, I decided I'd try to use machine learning to detect them.

After some literature reviews, I found Chicago hotel review dataset by Myle Ott and team which is considered to be a gold-standard dataset for research in this area. The dataset can be found [here](https://myleott.com/op-spam.html). It consists of 1,600 reviews divided into 50:50 fake and real, and 50:50 positive and negative where fake reviews are generated by hiring people through Amazon Mechanical Turk to write them. Myle’s model achieved 87% accuracy; this is used as my baseline performance. Although these are reviews for hotels, I consider it to sufficiently relate to the problem of interest.

Two models were developed: traditional NLP model using bag-of-word and lexicon features and fine-tuned pretrained neural network model. For the first model, data was cleaned by expanding word contractions, removing puncuations, tokenizing, lemmatizing, and removing stop words. Then unigram, bigram, and trigram bag-of-word features are created from this cleaned word tokens. Those features are then concatenated to other lexicon features - number of stop words, number of charactors per review, sentence length, number of first singular pronoun used - and fed into Random Forest which achieved 90% test set accuracy.

In the second model, state-of-the-art model in NLP - BERT - is adopted. Drop out and classification layer is added to the model, then the fine-tune process is done by feeding raw text data into the model. As recommended by BERT's author, fine-tune process is done on all layers of BERT. Although this could lead to a better result, the model could easily overfit the training data. Thus, a very small learning rate is used and we only train the model for 3 epochs. The model achieved 92% test accuracy, beating baseline by 5% absolute.

A research has shown that human accuracy on this task is around 50%, thus this model can definitely help evaluate trustworthiness of online reviews.

## Acknowledgements

 - [Paul, H., & Nikolaev, A. (2021). Fake review detection on online E-commerce platforms: a systematic literature review. Data Mining and Knowledge Discovery, 35(5), 1830-1881.](https://link.springer.com/article/10.1007/s10618-021-00772-6)
 - [Ott, M., Cardie, C., & Hancock, J. T. (2013, June). Negative deceptive opinion spam. In Proceedings of the 2013 conference of the north american chapter of the association for computational linguistics: human language technologies (pp. 497-501).](https://aclanthology.org/N13-1053.pdf)
 - [Ott, M., Choi, Y., Cardie, C., & Hancock, J. T. (2011). Finding deceptive opinion spam by any stretch of the imagination. arXiv preprint arXiv:1107.4557.](https://arxiv.org/abs/1107.4557)
 

