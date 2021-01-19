import os
import cv2 
import stanza
import speech_recognition
import time
import ffmpy
import sys
import pyautogui
import os.path
import shutil

from os import path
from PIL import Image
from difflib import SequenceMatcher
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from moviepy.editor import *

##
## Section 00	
## Function Definitions
## Improvements: 
##

## Improvements: Disable the keep or discard alert when downloading .swf files from chrome
## Improvements: Check lines 37-41 some letters don't change accurately
#Downloads Necessary Words From Online ASL Database
def DownloadWordTranslation(word,DictionaryDirectory):
	chrome_options = webdriver.ChromeOptions()
	prefs = {"safebrowsing.enabled": "False"}
	chrome_options.add_argument("--disable-extensions")
	chrome_options.add_experimental_option("prefs", prefs)
	
	browser = webdriver.Chrome(executable_path=ChromeDriverManager().install(),chrome_options = chrome_options)
	#browser.maximize_window()
	browser.get("http://www.aslpro.com/cgi-bin/aslpro/aslpro.cgi")
	first_letter = word[0]
	letters = browser.find_elements_by_xpath('//a[@class="sideNavBarUnselectedText"]')
	
	#proceed through list of letters on website to locate words with similar first letter as requested word. 
	for letter in letters:
		if first_letter == str(letter.text).strip().lower():
			letter.click()
			time.sleep(2)
			break

	# Show drop down menu ( Spinner )
	spinner = browser.find_elements_by_xpath("//option")
	best_score = -1.
	closest_word_item = None
	for item in spinner:
		item_text = item.text
		
		#proceed through list of words and compare to requested word 
		Similarity = SequenceMatcher(None, word.lower(), str(item_text).lower()).ratio()
		if Similarity > best_score:
			best_score = Similarity
			closest_word_item = item
			print(word, " ", str(item_text).lower())
			print("Score: " + str(Similarity))
	if best_score < AbsoluteSimilarityNecessary:
		print(word + " not found in dictionary")
		word = word + "\n"
		WordsNotInDatabaseFile = open("WordsNotInDatabase.txt", "a")
		WordsNotInDatabaseFile.write(word)
		WordsNotInDatabaseFile.close()
		browser.close()
		return
	real_name = str(closest_word_item.text).lower()

	#Download best matched word video file
	print("Downloading " + real_name + "...")
	closest_word_item.click()
	DownloadAccepted = False
	while DownloadAccepted == False:
		DownloadAccepted = AcceptFileDownload()
		print(DownloadAccepted)
		time.sleep(3)
	input_path = "C:\\Users\\jlcas\\Downloads\\" + real_name + ".swf"
	output_path = DictionaryDirectory + "\\" + real_name + ".mp4"
	
	#Convert video file from .swf to .mp4 format
	if(path.exists(output_path) != True):
		ff = ffmpy.FFmpeg(
		executable = "C:\\Users\\jlcas\\Lib\\site-packages\\ffmpeg.exe",
		inputs = {input_path: None},
		outputs = {output_path: None})
		ff.run()

	#Close Down Open Browsers After File is Downloaded
	browser.close()
	return real_name

def AcceptFileDownload():
	LocationOfDownloadAcceptance = pyautogui.locateOnScreen("KeepFileImage.png",grayscale = False, confidence = 0.90)
	if LocationOfDownloadAcceptance == None:
		return False
	else:
		pyautogui.click(pyautogui.center(LocationOfDownloadAcceptance))
		pyautogui.click(pyautogui.center(LocationOfDownloadAcceptance))
		time.sleep(2)
		return True

def CreateFingerSpellingImageVideo(RequestedWord,FingerSpellingImages,ASLImageDirectory,DictionaryDirectory):
	InputCharacters = list(RequestedWord)
	CurrentTranslation = []
	MappedValue = 0
	i = 0
	while(i <= len(InputCharacters) -1):
		#if a character is a letter
		if(InputCharacters[i].isalpha() == True):
			CurrentCharacter = InputCharacters[i].upper()
			MappedValue = ord(str(CurrentCharacter)) -  55
		#if a character is a number
		if(InputCharacters[i].isnumeric() == True):
			MappedValue = ord(str(InputCharacters[i])) - 48
		#if a character is a space
		if(InputCharacters[i] == " "):
			MappedValue = ord(str(InputCharacters[i])) + 4
		#if there is a different character that is necessary
		#
		
		#Ongoing Matrix of image paths in order
		CurrentTranslation.append(FingerSpellingImages[MappedValue])
		i = i+1

	#Compile Images into Video file
	ImageArray = []

	i = 0 
	while(i <= len(CurrentTranslation) -1):
		#Loads images and then resizes them into the same size
		Image = cv2.imread(os.path.join(ASLImageDirectory,CurrentTranslation[i]))
		Image = cv2.resize(Image,(260,240))
		Height, Width, Layers = Image.shape
		Size = (Width, Height)
		
		#sets the amount of seconds each picture is shown in the output video file
		time = 15
		#loads images into image array for conversion to video
		for j in range(time):
			ImageArray.append(Image)

		i = i+1
		
	#print(len(ImageArray))
	#print(ImageArray)

	#outputs the video as a .mp4 format
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	CreatedVideoName = RequestedWord + ".mp4"
	VideoFile = cv2.VideoWriter(CreatedVideoName,fourcc, 30, Size)

	#loads images into video constructor 
	i = 0
	while(i <= len(ImageArray)-1):
		VideoFile.write(ImageArray[i])
		i=i+1

	#releases pointer to VideoFile
	VideoFile.release()
	
	#Move Created Video File to Video Library Directory
	shutil.move(os.path.join(os.getcwd(),CreatedVideoName),os.path.join(DictionaryDirectory,CreatedVideoName))


BothVideoFormats = False
ASLConfirmation = input("Output Video as ASL instead of fingerspelling? Yes or No. ")

if (ASLConfirmation.lower() == "yes"):
	MixedFingerAndASLVideo = True
else:
	MixedFingerAndASLVideo = False



##
## Section 01	
## Grabbing Audio From Microphone and Converting to Text
## Improvements: Completed
##
print("\n\n\nSection #001")
print("Grabbing Audio From Microphone and Converting to Text.\n\n")


# obtain audio from the microphone
Recognizer = speech_recognition.Recognizer()

#attempt translation with Google Speech Recognition
CorrectTranslation = False
while(CorrectTranslation == False):
	try:
		# obtain audio from the microphone
		with speech_recognition.Microphone() as source:
			print("Say something!")
			audio = Recognizer.listen(source)
    
		# for testing purposes, we're just using the default API key
		# to use another API key, use `Recognizer.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
		InputText = Recognizer.recognize_google(audio)
		print("Google Speech Recognition thinks you said: " + InputText)
		
		#make sure the correct string is translated or repeat
		TranslationCheck = input("Is this the correct translation. Yes or No.")
		if (TranslationCheck.lower() == "yes"):
			CorrectTranslation = True
		else:
			CorrectTranslation = False
		
	except speech_recognition.UnknownValueError:
		print("Google Speech Recognition could not understand audio")
		Retry = input("Want To Try Again? Yes or No.")
		if(Retry.lower() == "yes"):
			CorrectTranslation == True
		else:
			sys.exit()
	except speech_recognition.RequestError as e:
		print("Could not request results from Google Speech Recognition service; {0}".format(e))
		Retry = input("Want To Try Again? Yes or No.")
		if(Retry.lower() == "yes"):
			CorrectTranslation == True
		else:
			sys.exit()
			
#Receive Text Input from Googles Speech to Text Translator 
#Dummy Input For Testing Purposes
#InputText = input("Provide text string to translate: ")
print("\nTranslating: ", InputText)


##
## Section 02	
## Tokenize Input Text and Remove Unnecessary Words
## Improvements: Revise the removed words to keep necessary words
##
print("\n\n\nSection #002")
print("Tokenize Input Text and Remove Unnecessary Characters.\n\n")

if(MixedFingerAndASLVideo == True) or (BothVideoFormats == True):
	#List of Words Not Available in the ASL Pro Website
	WordsNotInDatabase = []
	with open("WordsNotInDatabase.txt", "r") as b_file:
		for WordsNotFound in b_file:
			WordsNotInDatabase.append(WordsNotFound.strip())

	#download the enlgish text parser from stanfordnlp if needed
	#stanza.download('en')       # This downloads the English models for the neural pipeline

	#start annotating text with the pretrained stanford en_ewt word bank
	nlp = stanza.Pipeline('en', use_gpu = False) # This sets up a default neural pipeline in English
	#nlp = stanza.Pipeline('en', processors='tokenize,pos', treebank='en_ewt', use_gpu=True, pos_batch_size=3000)

	#store the parsed text into the doc variable as "word - sentence structure" format 
	doc = nlp(InputText)

	#output word and sentence structure pairs
	for i, sent in enumerate(doc.sentences):
		print("[Sentence {}]".format(i+1))
		for word in sent.words:
			print("{:12s}\t{:12s}\t{:6s}\t{:d}\t{:12s}".format(\
				  word.text, word.lemma, word.pos, word.head, word.deprel))
		print("")

	ASLEnglishString = ""
	WordsKeptForTranslation = []
	#Restructuring Attempt #2
	for i, sent in enumerate(doc.sentences):
		CurrentSentence = ""
		#Parse out necessary words in sentence into restructured sentence
		for word in sent.words:
			if (word.upos != "PUNCT") and (word.deprel != "punct"): 
				CurrentSentence = CurrentSentence + word.text + " "
				WordsKeptForTranslation.append(word.text)
		#Append restructured sentence to fully translated output sentence
		ASLEnglishString = ASLEnglishString + CurrentSentence + "\n"

	print(ASLEnglishString)
	print(WordsKeptForTranslation)


##
## Section 003	
## Initialize Path Variables for Downloading and Storing Video Files
## Improvements: Complete
##
print("\n\n\nSection #003")
print("Initialize Path Variables for Downloading and Storing Video Files.\n\n")


#Get Current Working Directory
MainWorkingDirectory = os.getcwd()

#Create a Folder "VideoLibrary" to Store Downloaded Video Files and Get Directory Path
DictionaryFolderName = "VideoLibrary"
if not os.path.exists(DictionaryFolderName):
	os.mkdir(DictionaryFolderName)

DictionaryDirectory = os.path.join(MainWorkingDirectory,DictionaryFolderName)
FingerSpellingImageDirectory = os.path.join(MainWorkingDirectory,"FingerSpellingImages")
FingerSpellingImages = os.listdir(FingerSpellingImageDirectory)


if(MixedFingerAndASLVideo == True) or (BothVideoFormats == True):
	##
	## Section 004	
	## Download Video Files of Translated Words From ASLPRO.com
	## Improvements: Check for Synonyms of Words for Translation in Video Database
	##
	print("\n\n\nSection #004")
	print("Download Video Files of Translated Words From ASLPRO.com .\n\n")


	#Load Previously Downloaded Words into Database by removing last 4 characters ".mp4"
	AllVideos = os.listdir(DictionaryDirectory)
	WordsInDatabase = [V[:-4] for V in AllVideos]

	#Cross Check Words for Translation with Stored Words and Download Words not found
	MatchedWords = []
	AbsoluteSimilarityNecessary = 0.90
	for TranslatedWord in WordsKeptForTranslation:
		BestSimilarityCoefficient = -1
		ClosestWordMatch = None
		
		#Check if word is Available in the ASLPro Directory
		WordNotAvailable = False
		for ExcludedWords in WordsNotInDatabase:
			if TranslatedWord.lower() ==  ExcludedWords.lower():
				WordNotAvailable = True
		
		#Word May Be Avialable check if already downloaded or download it. 
		if WordNotAvailable  == False:
			#Check If Required Word is Already Present in the Dictionary Directory
			for StoredWord in WordsInDatabase:
				Similarity = SequenceMatcher(None, TranslatedWord.lower(), StoredWord.lower()).ratio()
				if Similarity > BestSimilarityCoefficient:
					BestSimilarityCoefficient = Similarity
					ClosestWordMatch = StoredWord
			
			#Accept the Matched Dictionary Word if Similarity is above 90%
			if BestSimilarityCoefficient > AbsoluteSimilarityNecessary:
				MatchedWords.append(ClosestWordMatch)	
			#Attempt to Download A Video File of the Translated Word
			else:
				ClosestWordMatch = DownloadWordTranslation(TranslatedWord,DictionaryDirectory)
				if ClosestWordMatch != None:
					MatchedWords.append(ClosestWordMatch)
		
		#Word is not Available in ASLPro Directory
		elif WordNotAvailable  == True:
			#Check If Required Word is Already Present in the Dictionary Directory
			WordAlreadyExistsAsMP4 = False
			for StoredWord in WordsInDatabase:
				Similarity = SequenceMatcher(None, TranslatedWord.lower(), StoredWord.lower()).ratio()
				if Similarity > BestSimilarityCoefficient:
					BestSimilarityCoefficient = Similarity
					ClosestWordMatch = StoredWord
					WordAlreadyExistsAsMP4 = True
				else:
					WordAlreadyExistsAsMP4 = False
			#Create .mp4 file for the Requested Word out of Fingerspelling Images
			if WordAlreadyExistsAsMP4 == False:
				print('Attempting to Create mp4 for ', TranslatedWord)
				CreateFingerSpellingImageVideo(TranslatedWord,FingerSpellingImages,FingerSpellingImageDirectory,DictionaryDirectory)
				MatchedWords.append(TranslatedWord)
			

	##
	## Section 005	
	## Merge Video Files of Translated Words and play output translation video
	## Improvements: Completed.
	##
	print("\n\n\nSection #005")
	print("Merge Video Files of Translated Words and play output translation video .\n\n")


	#Gather All Necessary Video File Paths
	TranslatedVideoFilePaths = []
	for words in MatchedWords:
		FileName = words + ".mp4"
		FilePath = os.path.join(DictionaryDirectory,FileName)
		TranslatedVideoFilePaths.append(VideoFileClip(FilePath))
		
	#Concactenate All the Video Files in The Necessary Order
	TranslatedASLVideo = concatenate_videoclips(TranslatedVideoFilePaths, method= 'compose')
	TranslatedASLVideo.to_videofile("TranslatedVideo.mp4", fps = 30, remove_temp = False)
	TranslatedVideoPath = os.path.join(MainWorkingDirectory,"TranslatedVideo.mp4")

	#Ask User If We Should Play The Translated Video Now
	PlayVideo = input("Play Translated Video File Now ? Yes or No.")
	if (PlayVideo.lower() == "yes"):
		os.startfile(TranslatedVideoPath)
	else:
		print("Translated Video can be found at: ", TranslatedVideoPath)
		
if(MixedFingerAndASLVideo == False) or (BothVideoFormats == True):
	#Map characters to ASL images and organize images in appropriate sequence
	CurrentTranslation = []
	MappedValue = 0
	i = 0
	InputCharacters = list(InputText)
	while(i <= len(InputCharacters) -1):
		#if a character is a letter
		if(InputCharacters[i].isalpha() == True):
			CurrentCharacter = InputCharacters[i].upper()
			MappedValue = ord(str(CurrentCharacter)) -  55
			#if a character is a number
		if(InputCharacters[i].isnumeric() == True):
			MappedValue = ord(str(InputCharacters[i])) - 48
			#if a character is a space
		if(InputCharacters[i] == " "):
			MappedValue = ord(str(InputCharacters[i])) + 4
		#if there is a different character that is necessary
		#
									
		#Ongoing Matrix of image paths in order
		CurrentTranslation.append(FingerSpellingImages[MappedValue])
		i = i+1

	#Compile Images into Video file
	ImageArray = []

	i = 0 
	while(i <= len(CurrentTranslation) -1):
		#Loads images and then resizes them into the same size
		Image = cv2.imread(os.path.join(FingerSpellingImageDirectory,CurrentTranslation[i]))
		Image = cv2.resize(Image,(960,1600))
		Height, Width, Layers = Image.shape
		Size = (Width, Height)
								
		#sets the amount of seconds each picture is shown in the output video file
		Duration = 20
		#loads images into image array for conversion to video
		for j in range(Duration):
			ImageArray.append(Image)
		
		i = i+1
									
	#print(len(ImageArray))
	#print(ImageArray)

	#outputs the video as a .mp4 format
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	VideoFile = cv2.VideoWriter("TranslatedVideo.mp4",fourcc, 60, Size)

	#loads images into video constructor 
	i = 0
	while(i <= len(ImageArray)-1):
		VideoFile.write(ImageArray[i])
		i=i+1

	#releases pointer to VideoFile
	VideoFile.release()
	
	TranslatedVideoPath = os.path.join(MainWorkingDirectory,"TranslatedVideo.mp4")
	
	#Ask User If We Should Play The Translated Video Now
	PlayVideo = input("Play Translated Video File Now ? Yes or No.")
	if (PlayVideo.lower() == "yes"):
		os.startfile(TranslatedVideoPath)
	else:
		print("Translated Video can be found at: ", TranslatedVideoPath)