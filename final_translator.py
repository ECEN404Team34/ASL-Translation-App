from __future__ import division, print_function
import gzip, os, re
from math import log



__version__ = '2.0.0'

#Testing Code
from math import log
from keras.models import load_model
import sklearn as sk
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from tensorflow.keras.models import load_model



#loading the trained and validated model
current_model = load_model('Us_model.h5')


import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


#gathering test data and setting parameters
test_datagen = ImageDataGenerator(
        rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    'ushands/Test',                     #will need to change to our app
    batch_size = 1,                         #directory
    class_mode = 'categorical',
    target_size = (64,64),
    shuffle=False
    )


predictions = current_model.predict(test_gen)
predictions = np.argmax(predictions, axis=1)

a = 97 #ascii for character before a s.t. if image = a, pred = 1+96
#numpy array to letters
res = ""
predictions = [p + a for p in predictions]
for val in predictions:
    res = res + chr(val)

#2nd try
# I did not author this code, only tweaked it from:
# http://stackoverflow.com/a/11642687/2449774
# Thanks Generic Human!


# Modifications by Scott Randal (Genesys)
#
# 1. Preserve original character case after splitting
# 2. Avoid splitting every post-digit character in a mixed string (e.g. 'win32intel')
# 3. Avoid splitting digit sequences
# 4. Handle input containing apostrophes (for possessives and contractions)
#
# Wordlist changes:
# Change 2 required adding single digits to the wordlist
# Change 4 required the following wordlist additions:
#   's
#   '
#   <list of contractions>


class LanguageModel(object):
  def __init__(self, word_file):
    # Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).
    with gzip.open(word_file) as f:
      words = f.read().decode().split()
    self._wordcost = dict((k, log((i+1)*log(len(words)))) for i,k in enumerate(words))
    self._maxword = max(len(x) for x in words)


  def split(self, s):
    """Uses dynamic programming to infer the location of spaces in a string without spaces."""
    l = [self._split(x) for x in _SPLIT_RE.split(s)]
    return [item for sublist in l for item in sublist]


  def _split(self, s):
    # Find the best match for the i first characters, assuming cost has
    # been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).
    def best_match(i):
      candidates = enumerate(reversed(cost[max(0, i-self._maxword):i]))
      return min((c + self._wordcost.get(s[i-k-1:i].lower(), 9e999), k+1) for k,c in candidates)

    # Build the cost array.
    cost = [0]
    for i in range(1,len(s)+1):
      c,k = best_match(i)
      cost.append(c)

    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(s)
    while i>0:
      c,k = best_match(i)
      assert c == cost[i]
      # Apostrophe and digit handling (added by Genesys)
      newToken = True
      if not s[i-k:i] == "'": # ignore a lone apostrophe
        if len(out) > 0:
          # re-attach split 's and split digits
          if out[-1] == "'s" or (s[i-1].isdigit() and out[-1][0].isdigit()): # digit followed by digit
            out[-1] = s[i-k:i] + out[-1] # combine current token with previous token
            newToken = False
      # (End of Genesys addition)

      if newToken:
        out.append(s[i-k:i])

      i -= k

    return reversed(out)

DEFAULT_LANGUAGE_MODEL = LanguageModel(os.path.join(os.path.dirname(os.path.abspath(__file__)),'wordninja','wordninja_words.txt.gz'))
_SPLIT_RE = re.compile("[^a-zA-Z0-9']+")

def split(s):
  return DEFAULT_LANGUAGE_MODEL.split(s)

c = 'wethepeopleoftheunitedstatesinordertoformamoreperfectunionestablishjusticeinsuredomestictranquilityprovideforthecommondefencepromotethegeneralwelfareandsecuretheblessingsoflibertytoourselvesandourposteritydoordainandestablishthisconstitutionfortheunitedstatesofamerica'
d = 'WeholdthesetruthstobeselfevidentthatallmenarecreatedequalthattheyareendowedbytheirCreatorwithcertainunalienableRightsthatamongtheseareLifeLibertyandthepursuitofHappinessThattosecuretheserightsGovernmentsareinstitutedamongMenderivingtheirjustpowersfromtheconsentofthegovernedThatwheneveranyFormofGovernmentbecomesdestructiveoftheseendsitistheRightofthePeopletoalterortoabolishitandtoinstitutenewGovernmentlayingitsfoundationonsuchprinciplesandorganizingitspowersinsuchformastothemshallseemmostlikelytoeffecttheirSafetyandHappinessPrudenceindeedwilldictatethatGovernmentslongestablishedshouldnotbechangedforlightandtransientcausesandaccordinglyallexperiencehathshewnthatmankindaremoredisposedtosufferwhileevilsaresufferablethantorightthemselvesbyabolishingtheformstowhichtheyareaccustomedButwhenalongtrainofabusesandusurpationspursuinginvariablythesameObjectevincesadesigntoreducethemunderabsoluteDespotismitistheirrightitistheirdutytothrowoffsuchGovernmentandtoprovidenewGuardsfortheirfuturesecuritSuchhasbeenthepatientsufferanceoftheseColoniesandsuchisnowthenecessitywhichconstrainsthemtoaltertheirformerSystemsofGovernmentThehistoryofthepresentKingofGreatBritainisahistoryofrepeatedinjuriesandusurpationsallhavingindirectobjecttheestablishmentofanabsoluteTyrannyovertheseStatesToprovethisletFactsbesubmittedtoacandidworld'
r = 'HowdymynameisBrittanyPitcherandiamanelectricalengineeringmajorfromspringtxbutmostimportantlyiamtheloudestandproudestmemberofthefightingtexasaggieclassoftwentytwentyoneawhoop'
z = 'hellomynameisbrittanypitcherandmyfavoritecolorismarooniaminseniordesignrightnowthisiswhyiamworkingonthisprojectitismeanttohelpthosewhoarehardofhearingordeaftoovercomelanguagebarrierswiththeirpeersiamexcitedforittobefinishedandtodeterminghowwellitworks'

#create string of r, c, d
c = " ".join(split(c))
d = " ".join(split(d))
r = " ".join(split(r))
z = " ".join(split(z))



#try to split sentences

from nnsplit import NNSplit
splitter = NNSplit.load("en")

splits = splitter.split([res])[0]     

i = len(splits)-1
#split can be iterated over
for sentence in splits:
    print (sentence,end='')
    if(i>0):
        print("\b.")
        i = i-1
    else:
        print('.')
