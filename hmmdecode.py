import json
from collections import defaultdict
import sys
import math

class HiddenMarkovModel:
  def __init__(self, f):
    """
    tag_set: Set of known tags
    wordvstag: Calculates P(current tag | word)
    tagvstag: Calculates P(current tag | previous tag)
    """
    self.file_content = f
    self.tag_set = defaultdict(float)
    self.word_set = defaultdict(float)
    self.wordvstag = defaultdict(lambda: defaultdict(float))
    self.tagvsword = defaultdict(lambda: defaultdict(float))
    self.tagvstag = defaultdict(lambda: defaultdict(float))
    self.oc_tags = defaultdict(float)
  
  def train(self):
    for line in self.file_content:
      
      ### Add Start tag and End tag

      prevtag = 'Start';
      self.tag_set['Start'] += 1
      for word in line.split():
        w, t = word.rsplit("/", 1);

        self.tag_set[t]+=1
        
        ## Add frequency of w/t occurance
        self.wordvstag[w][t]+=1

        self.tagvsword[t][w]+=1
        
        ## Add frequency of prev tag/current tag occurance
        self.tagvstag[prevtag][t]+=1
        
        prevtag = t

    ## Mark the End token
    self.tag_set['End'] += 1
    self.tagvstag[prevtag]['End']+=1

    ## Transition probabilities + Add-one Smoothing
    for prevtag in self.tagvstag:
      for tag in self.tag_set:
        numerator = self.tagvstag[prevtag][tag] + 1
        denominator = self.tag_set[prevtag] + 1
        self.tagvstag[prevtag][tag] = math.log(numerator)  - math.log(denominator)

    ## Calculate the Emission probabilities
    for word in self.wordvstag:
      for tag in self.wordvstag[word]:
        self.wordvstag[word][tag] = math.log(self.wordvstag[word][tag]) - math.log(self.tag_set[tag])

    for tag in self.tagvsword:
      denom = 0
      for word in self.tagvsword[tag]:
        denom = denom + self.tagvsword[tag][word]
      
      for word in self.tagvsword[tag]:
        self.tagvsword[tag][word] = math.log(self.tagvsword[tag][word] / denom)
    
    # Check for Open Class tags
    for tag in self.tag_set:
      if len(self.tagvsword[tag]) > 0.10* len(self.wordvstag):
        self.oc_tags[tag]+=1
      


  def predict(self):
    result=[]

    for line in self.file_content:


      words = line.split()
      
      # Store the results
      prob_bp = {}  

      word = words[0]
      prob_bp[0] = {}
      if word in self.wordvstag:
                tagList = self.wordvstag[word]
                for tag in tagList:
                    prob_bp[0][tag] = {}
                    prob_bp[0][tag]['prob'] = self.wordvstag[word][tag] + self.tagvstag['Start'][tag]
                    prob_bp[0][tag]['bp'] = 'Start'
      else:
                tagList = self.tag_set
                for tag in self.oc_tags:
                    if tag != 'Start' and tag != 'End':
                        prob_bp[0][tag] = {}
                        prob_bp[0][tag]['prob'] = self.tagvstag['Start'][tag]
                        prob_bp[0][tag]['bp'] = 'Start'

      for i in range(1, len(words)):
        prob_bp[i] = {}
        word = words[i]
        if word in self.wordvstag:
            tagList = self.wordvstag[word]
            for tag in tagList:

                prob_bp[i][tag] = {}
                maxval = -sys.maxsize - 1
                bp = ''

                for prevtag in prob_bp[i - 1]:
                    prob = prob_bp[i - 1][prevtag]['prob'] + self.wordvstag[word][tag] + self.tagvstag[prevtag][tag]
                    if prob > maxval:
                        maxval = prob
                        bp = prevtag
                prob_bp[i][tag]['prob'] = maxval
                prob_bp[i][tag]['bp'] = bp
        else:
            tagList = self.tag_set
            for tag in self.oc_tags:
                if tag != 'Start' and tag != 'End':  
                    prob_bp[i][tag] = {}
                    maxval = -sys.maxsize - 1
                    bp = ''
                    for prevtag in prob_bp[i - 1]:
                        prob = prob_bp[i - 1][prevtag]['prob'] + self.tagvstag[prevtag][tag]
                        if prob > maxval:
                            maxval = prob
                            bp = prevtag
                    prob_bp[i][tag] = {}
                    prob_bp[i][tag]['prob'] = maxval
                    prob_bp[i][tag]['bp'] = bp
    
      i = len(words)
      prob_bp[i] = {}
      maxval = -sys.maxsize - 1
      bp = ''
      for prevtag in prob_bp[i - 1]:
          prob = prob_bp[i - 1][prevtag]['prob'] +  self.tagvstag[prevtag]['End']
          if prob > maxval:
                maxval = prob
                bp = prevtag
      prob_bp[i]['End'] = {}
      prob_bp[i]['End']['prob'] = maxval
      prob_bp[i]['End']['bp'] = bp
    
      
      tag = 'End'
      res = words[len(prob_bp)-2] +"/" + prob_bp[len(prob_bp)-1][tag]['bp']
      tag = prob_bp[len(prob_bp)-1][tag]['bp']
      for i in range(len(prob_bp)-2, 0, -1):
        tag = prob_bp[i][tag]['bp']
        res = words[i-1] + "/" + tag + " " + res
      
      result.append(res)
    
    return result


path = sys.argv[1]
f = open(path, encoding='UTF-8')
lines = f.readlines()
HMM_2 = HiddenMarkovModel(lines)
f = open('hmmmodel.txt', encoding='UTF-8')
model = json.loads(f.read())
HMM_2.tag_set = model[0]
HMM_2.tagvstag = model[1]
HMM_2.wordvstag = model[2]
HMM_2.oc_tags = model[3]

result = HMM_2.predict()

r = open('hmmoutput.txt', mode='w', encoding='UTF-8')
for line in result:
   r.write(line)
   r.write('\n')
r.close()
