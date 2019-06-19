from textblob import TextBlob
import glob
import os

neutralFactor = 0.0
posArray = []
negArray = []

posArraySubj = []
negArraySubj = []

fileCheck = open("CheckFile.txt", "w") #This is the check file
#dataFile =  open("2000_data.csv", "w") #This data file has been created to feed into the Machine learning model


for i in range(0,2):
	if(i%2 == 0):
		file_list = glob.glob(os.path.join(os.getcwd(), "TestCase2/pos", "*.txt")) #Mention the path here
		fileCheck.write("--------POSITIVE----------\r\n")
	else:
		file_list = glob.glob(os.path.join(os.getcwd(), "TestCase2/neg", "*.txt"))
		fileCheck.write("----------NEGATIVE----------\r\n")

	posCount = 0
	negCount = 0
	veryPosCount = 0
	veryNegCount = 0	
	neutralCount = 0

	#remove the encoding parameter if you're getting an error
	for file_path in file_list:
		with open(file_path, encoding="utf8") as f_input:
			text = f_input.read()
			blob = TextBlob(text)
			if(i%2 == 0):
				posArray.append(blob.sentiment.polarity)
				#dataFile.write("%3f,%3f,1 \n" % (blob.sentiment.polarity,blob.sentiment.subjectivity) )
				#posArraySubj.append(blob.sentiment.subjectivity)
			else:
				negArray.append(blob.sentiment.polarity)
				#dataFile.write("%3f,%3f,0 \n" % (blob.sentiment.polarity,blob.sentiment.subjectivity) )
				#negArraySubj.append(blob.sentiment.subjectivity)
			if blob.sentiment.polarity > 0.7:
				fileCheck.write("%s VeryPos %3f\r\n" % ((file_path.split('\\')[5]), blob.sentiment.polarity))
				veryPosCount+=1
			elif blob.sentiment.polarity > 0.15:
				fileCheck.write("%s Pos %3f\r\n" % ((file_path.split('\\')[5]), blob.sentiment.polarity))
				posCount+=1
			elif blob.sentiment.polarity < 0.05:
				fileCheck.write("%s Negative %3f\r\n" % ((file_path.split('\\')[5]), blob.sentiment.polarity))
				negCount+=1
			elif blob.sentiment.polarity < -0.7:
				fileCheck.write("%s VeryNegative %3f\r\n" % ((file_path.split('\\')[5]), blob.sentiment.polarity))
				veryNegCount+=1	
			else:
				fileCheck.write("%s Neutral %3f\r\n" % ((file_path.split('\\')[5]), blob.sentiment.polarity))
				neutralCount+=1				
	if(i%2 == 0):			
		print("For Positive Tests")
	else:
		print("For Negative Tests")
	print(" VeryPos ",veryPosCount," Positive ",posCount," Negative ", negCount," veryNegCount ", veryNegCount, " Neutral ", neutralCount)













#Viusualizing the results
# import matplotlib.pyplot as  plt
# l1, = plt.plot(posArray,'b')
# l2, = plt.plot(negArray,'r')
# # l3, = plt.plot(posArraySubj,'g')
# # l4, = plt.plot(negArraySubj,'y')
# plt.legend(['posArray','negArray'], loc='upper left')
# plt.show()

# import matplotlib.pyplot as  plt
# l1, = plt.plot(posArraySubj,'b')
# l2, = plt.plot(negArraySubj,'r')
# plt.legend(['posArray','negArray'],loc='upper left')
# plt.show()

# plt.plot(posArray)
# plt.show()

# plt.plot(negArray,'b')
# plt.show()


#this is the code to run TextBlob using NaiveBayesAnalyzer

#from textblob import TextBlob
# from textblob.sentiments import NaiveBayesAnalyzer

# print("")
# print("NaiveBayesAnalyzer")
# print("")

# for i in range(0,2):
# 	if(i%2 == 0):
# 		file_list = glob.glob(os.path.join(os.getcwd(), "posTest", "*.txt"))
# 	else:
# 		file_list = glob.glob(os.path.join(os.getcwd(), "negTest", "*.txt"))

# 	posCount = 0
# 	negCount = 0	
# 	neutralCount = 0
# 	count = 0
# 	for file_path in file_list:
# 		with open(file_path) as f_input:
# 			text = f_input.read()
# 			print(count)
# 			count+=1
# 			blob = TextBlob(text, analyzer=NaiveBayesAnalyzer())
# 			if blob.sentiment.p_pos > 0.5:
# 				posCount+=1
# 			elif blob.sentiment.p_neg > 0.5:
# 				negCount+=1
# 			else:
# 				neutralCount+=1
# 			if(count%20 == 0):
# 				print("Positive ",posCount," Negative ", negCount, " Neutral ", neutralCount)				
# 	if(i%2 == 0):			
# 		print("For Positive Tests")
# 	else:
# 		print("For Negative Tests")
# 	print("Positive ",posCount," Negative ", negCount, " Neutral ", neutralCount)

# blob = blob.correct()

# Just intsall textblob on your machine using the command: pip install textblob

# Polarity is float which lies in the range of [-1,1] where 1 means positive statement and -1 means a negative statement. 
# Subjective sentences generally refer to personal opinion, emotion or judgment whereas objective refers to factual information. 
# subjectivity is also a float which lies in the range of [0,1].

#text = "Just love the X. Feel so Premium and a Head turner too. Face ID working fine but still miss " + "the fingerprint scanner very much. I jump from 5S to X so it’s a huge skip. I’m very very happy"+ " with it. Specially battery backup is great after using with 4g cellular network and no heating "+ "issue at all, though I’m not a mobile gamer, Oftentimes I play Boom Beach and I watch YouTube "+ "videos and I surf a lot. It makes a deep hole in pocket at the Hefty price tag. So it’s all "+ "upto your Consideration"

