#import libraries which are needed
import nltk
import random
import feedparser
#create a dict called urls which contains list of url of rss feed
urls = {
  'u1': 'any rss feed url,
  'u2': 'any rss feed url',
}
feedmap = {}
stopwords = nltk.corpus.stopwords.words('english')
#This function, takes list of words and then adds them to the dictionary,
#where each key is the word and the value is True.
def featureExtractor(words):
  features = {}
  for word in words:
    if word not in stopwords:
      features["word({})".format(word)] = True
    return features
sentences = []
for category in urls.keys():
    feedmap[category] = feedparser.parse(urls[category])
    print("downloading {}".format(urls[category]))
    for entry in feedmap[category]['entries']:
        data = entry['summary']
        words = data.split()
        sentences.append((category, words))
featuresets = [(featureExtractor(words), category) for category, words in sentences]
random.shuffle(featuresets)
#this step to split the data for test and train dataset
total = len(featuresets)
off = int(total/2)
trainset = featuresets[off:]
testset = featuresets[:off]
#Create a classifier using the trainset data by using the NaiveBayesClassifier module's train() function
classifier = nltk.NaiveBayesClassifier.train(trainset)
print(nltk.classify.accuracy(classifier, testset))
#Print the informative features about this data using the built-in function of classifier:
classifier.show_most_informative_features(5)
#Take four sample entries from the u1 RSS item. Try to tag the document based on title
for (i, entry) in enumerate(feedmap['u1']['entries']):
  if i < 4:
    features = featureExtractor(entry['title'].split())
    category = classifier.classify(features)
    print('{} -> {}'.format(category, entry['title']))
