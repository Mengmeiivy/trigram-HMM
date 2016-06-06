import numpy as np

def is_number(word):
	try:
		float(word)
		return True
	except ValueError:
		return False

def get_prior (prevtwotag, prevtag, tag):
	lambda1 = 0.1
	lambda2 = 0.5 
	lambda3 = 1-lambda1-lambda2

	if tag == 'END':
		uni_prior = 0.0
		lambda1 = 0
		lambda2 = 0.6
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

def get_emission (word, tag, twotag, notfirst):
	first = word[0]
	one = word[-1]
	emission = 0.0

	if is_number(word):
		for entry in wordprob:
			if (is_number(entry)) and (tag in wordprob[entry]):
				emission += 1
		#emission = emission/tagprob[tag]['UNI']
		emission = emission/tagprob[tag]['SUM']		

	elif len(word) >= 2:
		if (first.isupper() and notfirst):
			for entry in wordprob:
				if (wordprob[entry]['SUM'] < 10):
					if (entry[0].isupper()) and (tag in wordprob[entry]):
						emission += 1
					if (entry[-1] == one) and (tag in wordprob[entry]):
						emission += 1
			#emission = emission*0.5/tagprob[tag]['UNI']
			emission = emission/tagprob[tag]['SUM']		
		else:
			if len(word) >= 4:
				two = word[-2:]
				three = word[-3:]
				four = word[-4:]
				for entry in wordprob:
					if (wordprob[entry]['SUM'] < 10):
						if (entry[-1] == one) and (tag in wordprob[entry]):
							emission += 1*0.3
						if (entry[-2:] == two) and (tag in wordprob[entry]):
							emission += 1*0.3
						if (entry[-3:] == three) and (tag in wordprob[entry]):
							emission += 1*0.3
						if (entry[-4:] == four) and (tag in wordprob[entry]):
							emission += 1*0.1
				#emission = emission/tagprob[tag]['UNI']
				emission = emission/tagprob[tag]['SUM']		

	if emission == 0.0 and not is_number(word):
		emission = 0.5*(unknownprob.get(tag, 0.0) + unknownprob.get(twotag, 0.0))
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
			#emission = 0.5*(unknownprob.get(tag, 0.0) + unknownprob.get(twotag, 0.0))
			emission = get_emission(sent[0], tag, twotag, False)
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
					#emission = 0.5*(unknownprob.get(tag, 0.0) + unknownprob.get(twotag, 0.0))
					emission = get_emission(sent[1], tag, twotag, True)
				viterbi[q, 2, p] = viterbi[p, 1, 0]*prior*emission

	if (length2 > 4):
		for i in range (3, length2-1):
			for q in range (length1-1):
				tag = taglist[q]
				for p in range (length1-1):
					prevtag = taglist[p]
					twotag = prevtag + tag
					if (sent[i-1] in wordprob):
						emission = wordprob[sent[i-1]].get(tag, 0.0)
					else:
						#emission = 1/100000
						#emission = 0.5*(unknownprob.get(tag, 0.0) + unknownprob.get(twotag, 0.0))
						emission = get_emission(sent[i-1], tag, twotag, True)
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
f = open('WSJ_02-21.pos', 'r')
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
f1 = open('WSJ_24.pos', 'r')
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
f2 = open('WSJ_23.words', 'r')
#f2 = open('medium.txt', 'r')
output = open('english_output.txt', 'w')
#output = open('medium_output.txt', 'w')
sent = []
for line in f2:
	if (line != '\n'):
		sent.append(line.strip('\n'))
	else:
		print (sent)
		viterbi (sent, wordprob, tagprob, taglist)
		sent = []














