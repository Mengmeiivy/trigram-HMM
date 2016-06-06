import numpy as np

def hasDigit(a):
	for i in a:
		if i.isdigit():
			return True
	return False

def number1(a):
	number1_list = ['.', '%']
	number1_list.append("分")
	number1_list.append("十")
	number1_list.append("百")
	number1_list.append("千")
	number1_list.append("万")
	number1_list.append("亿")
	number1_list.append("多")

	for i in a:
		if (not i.isdigit()) and (i not in number1_list):
			return False
	return True

def number2(a):
	number1_list = []
	number1_list.append("月")
	number1_list.append("份")
	number1_list.append("年")
	number1_list.append("天")
	number1_list.append("月")
	number1_list.append("旬")
	number1_list.append("号")

	if a[-1] in number1_list:
		return True
	else:
		return False

def get_prior (prevtwotag, prevtag, tag):
	lambda1 = 0.1
	lambda2 = 0.4 
	lambda3 = 1-lambda1-lambda2

	if tag == 'END':
		uni_prior = 0.0
		lambda1 = 0
		lambda2 = 0.5
		lambda3 = 1-lambda1-lambda2
	else:
		uni_prior = tagprob[tag]['UNI']

	bi_prior = tagprob[prevtag].get(tag, 0.0)

	if prevtwotag not in tagprob:
		tri_prior = 0.0
	else:
		tri_prior = tagprob[prevtwotag].get(tag, 0.0)

	prior = lambda1*uni_prior + lambda2*bi_prior + lambda3*tri_prior

	return prior 

def get_emission (word, tag):

	emission = 0.0

	if hasDigit(word):
		if number1(word):
			for entry in wordprob:
				if (wordprob[entry]['SUM'] < 10):
					if (number1(entry)) and (tag in wordprob[entry]):
						emission += 1
		elif number2(word):
			for entry in wordprob:
				if (wordprob[entry]['SUM'] < 10):
					if (number2(entry)) and (tag in wordprob[entry]):
						emission += 1
		else:
			for entry in wordprob:
				if (wordprob[entry]['SUM'] < 10):
					if (hasDigit(entry)) and (tag in wordprob[entry]):
						emission += 1
	else:
		if len(word) == 2:
			one = word[-1]
			for entry in wordprob:
				if (wordprob[entry]['SUM'] < 10):
					if (entry[-1] == one) and (tag in wordprob[entry]):
						emission += 1
			
		elif len(word) >= 3:
			one = word[-1]
			two = word[-2:]
			for entry in wordprob:
				if (wordprob[entry]['SUM'] < 10):
					if (entry[-1] == one) and (tag in wordprob[entry]):
						emission += 1*0.5
					if (entry[-2:] == two) and (tag in wordprob[entry]):
						emission += 1*0.5

	emission = emission/tagprob[tag]['SUM']				

	if emission == 0.0:
		emission = unknownprob.get(tag, 0.0)
	return emission

#the viterbi algorithm
def viterbi (sent, wordprob, tagprob, taglist):

	length1 = len(taglist) + 1
	length2 = len(sent) + 2

	viterbi = np.zeros((length1, length2, length1))

	backpointer = np.empty((length1, length2, length1), dtype = object)
	answertag = []

	for q in range (length1 - 1):
		tag = taglist[q]
		twotag = 'START' + tag
		prior = tagprob['START'].get(tag, 0.0)
		if (sent[0] in wordprob):
			emission = wordprob[sent[0]].get(tag, 0.0)
		else:
			#emission = 1/100000
			#emission = unknownprob.get(tag, 0.0) #this is better than averaged 
			#emission = 0.5*(unknownprob.get(tag, 0.0) + unknownprob.get(twotag, 0.0))
			emission = get_emission(sent[0], tag)
		viterbi[q, 1, 0] = prior*emission

	if (length2 > 3):
		for q in range (length1 - 1):
			for p in range (length1 - 1):
				prevtag = taglist[p]
				tag = taglist[q]
				prevtwotag = 'START' + prevtag
				twotag = prevtag + tag
				prior = get_prior(prevtwotag, prevtag, tag)

				if (sent[1] in wordprob):
					emission = wordprob[sent[1]].get(tag, 0.0)
				else: 
					#emission = 1/100000
					#emission = unknownprob.get(tag, 0.0)
					#emission = 0.5*(unknownprob.get(tag, 0.0) + unknownprob.get(twotag, 0.0))
					emission = get_emission(sent[1], tag)
				viterbi[q, 2, p] = viterbi[p, 1, 0]*prior*emission

	if (length2 > 4):
		for i in range (3, length2-1):
			#print ('word number is', i)
			for q in range (length1-1):
				tag = taglist[q]
				for p in range (length1-1):
					prevtag = taglist[p]
					twotag = prevtag + tag
					if (sent[i-1] in wordprob):
						emission = wordprob[sent[i-1]].get(tag, 0.0)
					else:
						#emission = 1/100000
						#emission = unknownprob.get(tag, 0.0)
						#emission = 0.5*(unknownprob.get(tag, 0.0) + unknownprob.get(twotag, 0.0))
						emission = get_emission(sent[i-1], tag)
					maxscore = 0.0
					maxindex = 0
					for w in range (length1-1):
						prevprevtag = taglist[w]
						prevtwotag = prevprevtag + prevtag
						prior = get_prior(prevtwotag, prevtag, tag)
						currentmax = viterbi[p, i-1, w]*prior*emission 
						if (currentmax > maxscore):
							maxscore = currentmax
							maxindex = w
					viterbi[q, i, p] = maxscore
					backpointer[q, i, p] = maxindex


	maxscore = 0.0
	maxindex = 0
	tag = 'END'
	for p in range (length1-1):
		prevtag = taglist[p]
		if (length2 > 3):
			for w in range (length1-1):
				prevprevtag = taglist[w]
				prevtwotag = prevprevtag + prevtag
				prior = get_prior(prevtwotag, prevtag, tag)
				currentmax = viterbi[p, length2-2, w]*prior 
				if (currentmax > maxscore):
					maxscore = currentmax
					maxindex = w
			viterbi[length1-1, length2-1, p] = maxscore
			backpointer[length1-1, length2-1, p] = maxindex

		else: 
			prevprevtag = 'START'
			prevtwotag = prevprevtag + prevtag
			prior = get_prior(prevtwotag, prevtag, tag)
			currentmax = viterbi[p, 1, 0]*prior
			viterbi[length1-1, length2-1, p] = currentmax



	if (length2 > 3):
		maxscore = 0.0
		index1 = 0
		for p in range (length1-1):
			if viterbi[length1-1, length2-1, p] > maxscore:
				maxscore = viterbi[length1-1, length2-1, p]
				index1 = p 
		tag = taglist[index1]
		answertag.append(tag)
		index2 = backpointer[length1-1, length2-1, index1]
		tag = taglist[index2]
		answertag.append(tag)

		for i in range (length2-2, 2, -1):
			temp = index2
			index2 = backpointer[index1, i, index2]
			tag = taglist[index2]
			answertag.append(tag)
			index1 = temp

		answertag.reverse()

	else:
		maxscore = 0.0
		index = 0
		for p in range (length1-1):
			if viterbi[length1-1, length2-1, p] > maxscore:
				maxscore = viterbi[length1-1, length2-1, p]
				index = p 
		tag = taglist[index]
		answertag.append(tag)

	for i in range (len(sent)):
		output.write(sent[i] + '\t' + answertag[i] + '\n')
	output.write('\n')


"""-----the main program-----"""

wordprob = {}
tagprob = {}
tagprob['START'] = {}
unknownprob = {}
taglist = []
total_tag = 0

#open the training file
f = open('chinese_training.txt', 'r')
#f = open('small.txt', 'r')
prevtag = 'START'
prevtwotag = 'START'
for line in f:
	if (line != '\n'):
		total_tag += 1
		string = line.split()
		word = string[0]
		tag  = string[1]
		twotag = prevtag + tag 

		if tag not in taglist:
			taglist.append(tag)

		if word not in wordprob:
			wordprob[word] = {}
			wordprob[word][tag] = 1.0
			wordprob[word][twotag] = 1.0
			wordprob[word]['SUM'] = 1

		else:
			wordprob[word][tag] = wordprob[word].get(tag, 0.0) + 1.0
			wordprob[word][twotag] = wordprob[word].get(twotag, 0.0) + 1.0
			wordprob[word]['SUM'] += 1

		if tag not in tagprob:
			tagprob[tag] = {}
		tagprob[prevtag][tag] = tagprob[prevtag].get(tag, 0.0) + 1	
		tagprob[prevtag]['SUM'] = tagprob[prevtag].get('SUM', 0.0) + 1.0

		if prevtwotag != 'START':
			tagprob[prevtwotag][tag] = tagprob[prevtwotag].get(tag, 0.0) + 1.0
			tagprob[prevtwotag]['SUM'] = tagprob[prevtwotag].get('SUM', 0.0) + 1.0

		if twotag not in tagprob:
			tagprob[twotag] = {}

		prevtag = tag
		prevtwotag = twotag 
	else:
		tagprob[prevtag]['END'] = tagprob[prevtag].get('END', 0.0) + 1.0
		tagprob[prevtag]['SUM'] = tagprob[prevtag].get('SUM', 0.0) + 1.0
		tagprob[prevtwotag]['END'] = tagprob[prevtwotag].get('END', 0.0) + 1.0
		tagprob[prevtwotag]['SUM'] = tagprob[prevtwotag].get('SUM', 0.0) + 1.0
		prevtag = 'START'
		prevtwotag = 'START'


#open the development file
f1 = open('chinese_dev_pos.txt', 'r')
prevtag = 'START'
prevtwotag = 'START'
for line in f1:
	if (line != '\n'):
		total_tag += 1
		string = line.split()
		word = string[0]
		tag  = string[1]
		twotag = prevtag + tag 

		if tag not in taglist:
			taglist.append(tag)

		if word not in wordprob:
			wordprob[word] = {}
			wordprob[word][tag] = 1.0
			wordprob[word][twotag] = 1.0
			wordprob[word]['SUM'] = 1

		else:
			wordprob[word][tag] = wordprob[word].get(tag, 0.0) + 1.0
			wordprob[word][twotag] = wordprob[word].get(twotag, 0.0) + 1.0
			wordprob[word]['SUM'] += 1

		if tag not in tagprob:
			tagprob[tag] = {}
		tagprob[prevtag][tag] = tagprob[prevtag].get(tag, 0.0) + 1	
		tagprob[prevtag]['SUM'] = tagprob[prevtag].get('SUM', 0.0) + 1.0

		if prevtwotag != 'START':
			tagprob[prevtwotag][tag] = tagprob[prevtwotag].get(tag, 0.0) + 1.0
			tagprob[prevtwotag]['SUM'] = tagprob[prevtwotag].get('SUM', 0.0) + 1.0

		if twotag not in tagprob:
			tagprob[twotag] = {}

		prevtag = tag
		prevtwotag = twotag 
	else:
		tagprob[prevtag]['END'] = tagprob[prevtag].get('END', 0.0) + 1.0
		tagprob[prevtag]['SUM'] = tagprob[prevtag].get('SUM', 0.0) + 1.0
		tagprob[prevtwotag]['END'] = tagprob[prevtwotag].get('END', 0.0) + 1.0
		tagprob[prevtwotag]['SUM'] = tagprob[prevtwotag].get('SUM', 0.0) + 1.0
		prevtag = 'START'
		prevtwotag = 'START'


#process the likelihood dictionaries 
for word in wordprob:
	if (wordprob[word]['SUM'] == 1):
		for tag in wordprob[word]:
			if (tag != 'SUM'):
				unknownprob[tag] = unknownprob.get(tag, 0.0) + 1.0
	for tag in wordprob[word]:
		if (tag != 'SUM'):
			wordprob[word][tag] = wordprob[word][tag] / tagprob[tag]['SUM']
for tag in tagprob:
	for tag2 in tagprob[tag]:
		if (tag2 != 'SUM'):
			tagprob[tag][tag2] = tagprob[tag][tag2] / tagprob[tag]['SUM']
for tag in tagprob:
	tagprob[tag]['UNI'] = tagprob[tag]['SUM'] / total_tag
for tag in unknownprob:
	unknownprob[tag] = unknownprob[tag] / tagprob[tag]['SUM']

#open the test file and produce the output 
#f2 = open('small_c.txt', 'r')
#f2 = open('big_c.txt', 'r')
f2 = open('chinese_test_words.txt', 'r')
output = open('chinese_output.txt', 'w')
sent = []
for line in f2:
	if (line != '\n'):
		sent.append(line.strip('\n'))
	else:
		print (sent)
		viterbi (sent, wordprob, tagprob, taglist)
		sent = []
















