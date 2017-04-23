import csv

# Read bucket file
buckets = dict()
maxn = 0
with open('age_gender_bkts.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	reader.next()
	for row in reader:
		# Corresponding bucket
		if not (row[0], row[2]) in buckets:
			buckets[(row[0], row[2])] = [(row[1],float(row[3]))]
			maxn = max(maxn, float(row[3]))
		else:
			buckets[(row[0], row[2])].append((row[1],float(row[3])))
			maxn = max(maxn, float(row[3]))

# Scale
scaledbuckets = dict()
for k,v in buckets.iteritems():
	scaledbuckets[k] = []
	for v2 in sorted(v, key=lambda x: x[0]):
		scaledbuckets[k].append((v2[0], v2[1] / maxn * 2 - 1))

# Age buckets
agebuckets = dict()
for k,v in buckets.iteritems():
	if not k[0] in agebuckets:
		agebuckets[k[0]] = dict()
	for t in v:
		if not t[0] in agebuckets[k[0]]:
			agebuckets[k[0]][t[0]] = 0
		agebuckets[k[0]][t[0]] += t[1]

maxna = 0
minna = float('inf')
for v in agebuckets.values():
	maxna = max(maxna, max(v.values()))
	minna = min(minna, min(v.values()))

# Scale
scaledagebuckets = dict()
for k,v in agebuckets.iteritems():
	scaledagebuckets[k] = []
	for v2 in sorted(v.items(), key=lambda x: x[0]):
		scaledagebuckets[k].append((v2[0], (v2[1] - minna) / (maxna - minna) * 2 - 1))

# Gender buckets
genderbuckets = dict()
for k,v in buckets.iteritems():
	if not k[1] in genderbuckets:
		genderbuckets[k[1]] = dict()
	for t in v:
		if not t[0] in genderbuckets[k[1]]:
			genderbuckets[k[1]][t[0]] = 0
		genderbuckets[k[1]][t[0]] += t[1]

maxng = 0
minng = float('inf')
for v in genderbuckets.values():
	maxng = max(maxng, max(v.values()))
	minng = min(minng, min(v.values()))

# Scale
scaledgenderbuckets = dict()
for k,v in genderbuckets.iteritems():
	scaledgenderbuckets[k] = []
	for v2 in sorted(v.items(), key=lambda x: x[0]):
		scaledgenderbuckets[k].append((v2[0], (v2[1] - minng) / (maxng - minng) * 2 - 1))

# Generic buckets
genericbucket = dict()
for k,v in buckets.iteritems():
	for t in v:
		if not t[0] in genericbucket:
			genericbucket[t[0]] = 0
		genericbucket[t[0]] += t[1]

maxnb = max(genericbucket.values())
minnb = min(genericbucket.values())

# Scale
scaledgenericbucket = []
for v2 in sorted(genericbucket.items(), key=lambda x: x[0]):
	scaledgenericbucket.append((v2[0], (v2[1] - minnb) / (maxnb - minnb) * 2 - 1))

def getagebucket(age):
	if age == -1 or age is None or age == '':
		return None
	elif age < 5:
		return '0-4'
	elif age < 10:
		return '5-9'
	elif age < 15:
		return '10-14'
	elif age < 20:
		return '15-19'
	elif age < 25:
		return '20-24'
	elif age < 30:
		return '25-29'
	elif age < 35:
		return '30-34'
	elif age < 40:
		return '35-39'
	elif age < 45:
		return '40-44'
	elif age < 50:
		return '45-49'
	elif age < 55:
		return '50-54'
	elif age < 60:
		return '55-59'
	elif age < 65:
		return '60-64'
	elif age < 70:
		return '65-69'
	elif age < 75:
		return '70-74'
	elif age < 80:
		return '75-79'
	elif age < 85:
		return '80-84'
	elif age < 90:
		return '85-89'
	elif age < 95:
		return '90-94'
	elif age < 100:
		return '95-99'
	else:
		return '100+'

def getgenderbucket(gender):
	if gender == 'MALE':
		return 'male'
	elif gender == 'FEMALE':
		return 'female'
	else:
		return None

def getbucket(age, gender):
	agebucket = getagebucket(age)
	genderbucket = getgenderbucket(gender)
	
	onlyageknown = 1 if agebucket is not None and genderbucket is None else -1
	onlygenderknown = 1 if genderbucket is not None and agebucket is None else -1
	bothknown = 1 if genderbucket is not None and agebucket is not None else -1
	noneknown = 1 if agebucket is None and genderbucket is None else -1
	
	if bothknown == 1:
		datamap = map(lambda x: x[1], scaledbuckets[(agebucket, genderbucket)])
	elif onlyageknown == 1:
		datamap = map(lambda x: x[1], scaledagebuckets[agebucket])
	elif onlygenderknown == 1:
		datamap = map(lambda x: x[1], scaledgenderbuckets[genderbucket])
	else:
		datamap = map(lambda x: x[1], scaledgenericbucket)
	knowledgemap = [onlyageknown, onlygenderknown, bothknown, noneknown]
	
	return datamap + knowledgemap