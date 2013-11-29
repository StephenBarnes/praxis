#!/usr/bin/env python

from scipy import stats
from numpy import array

NUMBERS = tuple((i for i in "0123456789."))

class ProbabilityDist(object):
	def __init__(self):
		self.mean = 0.
		self.std = 1.
	def mean(self, VariableDefs, QuantValues, ChangedVariables, **kwargs):
		return self.mean
	def sqmean(self, VariableDefs, QuantValues, ChangedVariables):
		return self.var() + self.mean()**2
	def var(self, VariableDefs, QuantValues, ChangedVariables):
		return self.var
	def std(self, VariableDefs, QuantValues, ChangedVariables):
		return self.var(VariableDefs, QuantValues, ChangedVariables)**.5 #defining it this way round because variance has more straightforward rules, eg the variance of the sum is the sum of the variances, assuming independence
	def rvs(self):
		return None
	def rvs(self, n):
		return array([self.rvs() for i in xrange(n)])
	def __str__(self):
		return str(type(self))
	def __add__(self, Second):
		return MakeDistSum(self, Second)
	def __mul__(self, Second):
		return MakeDistProd(self, Second)
	def __sub__(self, Second):
		return self + DegenerateDist(-1.)*Second

class DistSum(ProbabilityDist):
	def __init__(self, Dist1=None, Dist2=None):
		self.left = Dist1
		self.right = Dist2
		#self.meanmemo = {}
		self.varmemo = None
	def rustle(self):
		self.meanmemo = None
		self.varmemo = None
	def mean(self, VariableDefs, QuantValues, ChangedVariables, **kwargs):
		#if self.meanmemo[(ChangedVariables,kwargs)] is None:
		#	self.meanmemo[(ChangedVariables,kwargs)] = self.left.mean(VariableDefs, QuantValues, ChangedVariables=ChangedVariables, **kwargs) + self.right.mean(VariableDefs, QuantValues, ChangedVariables=ChangedVariables, **kwargs)
		#return self.meanmemo[(ChangedVariables,kwargs)]
		return self.left.mean(VariableDefs, QuantValues, ChangedVariables, **kwargs) + self.right.mean(VariableDefs, QuantValues, ChangedVariables, **kwargs)
	def var(self, ChangedVariables={}):
		if self.varmemo is None:
			self.varmemo = self.left.var() + self.right.var() #assuming independence
		return self.varmemo
	def rvs(self, n=1, ChangedVariables={}):
		L = self.left.rvs(n)
		R = self.right.rvs(n)
		return array([L[i] + R[i] for i in xrange(n)])
	def __str__(self):
		R = str(type(self))
		for i in (self.left, self.right):
			itext = "\n" + str(i)
			R += itext.replace("\n","\n\t")
		return R
	def representatives(self, VariableDefs, QuantValues, ChangedVariables, **kwargs):
		R = {}
		for repleft, repleftweight in self.left.representatives().items():
			for repright, reprightweight in self.right.representatives().items():
				newrep = repleft + repright
				newrepweight = repleftweight * reprightweight
				if newrep in R:
					R[newrep] += newrepweight
				else:
					R[newrep] = newrepweight
		return R

class DistProd(ProbabilityDist):
	def __init__(self, Dist1=None, Dist2=None):
		self.left = Dist1
		self.right = Dist2
		self.varmemo = None
		self.meanmemo = None
	def rustle(self):
		self.varmemo = None
		self.meanmemo = None
	def mean(self, VariableDefs, QuantValues, ChangedVariables={}, **kwargs):
		#if self.meanmemo is None:
		#	self.meanmemo = self.left.mean(VariableDefs, QuantValues, ChangedVariables=ChangedVariables, **kwargs) * self.right.mean(VariableDefs, QuantValues, ChangedVariables=ChangedVariables, **kwargs) #assuming independence
		#print "mean of DistProd is:", self.meanmemo
		#return self.meanmemo
		return self.left.mean(VariableDefs, QuantValues, ChangedVariables=ChangedVariables, **kwargs) * self.right.mean(VariableDefs, QuantValues, ChangedVariables=ChangedVariables, **kwargs) #assuming independence
	def var(self, ChangedVariables={}):
		if self.varmemo is None:
			self.varmemo = sqmean(self, VariableDefs, QuantValues.left)*sqmean(self.right) - (self.left.mean()**2)*(self.right.mean()**2) #assuming independence
		return self.varmemo
		#FIXME for the case where one distribution is the degenerate distribution? or just leave it like this
	def rvs(self, n=1, ChangedVariables={}):
		L = self.left.rvs(n)
		R = self.right.rvs(n)
		return array([L[i] * R[i] for i in xrange(n)])
	def __str__(self):
		R = str(type(self))
		for i in (self.left, self.right):
			itext = "\n" + str(i)
			R += itext.replace("\n","\n\t")
		return R
	def representatives(self, VariableDefs, QuantValues, ChangedVariables, **kwargs):
		R = {}
		for repleft, repleftweight in self.left.representatives(VariableDefs, QuantValues, ChangedVariables, **kwargs).items():
			for repright, reprightweight in self.right.representatives(VariableDefs, QuantValues, ChangedVariables, **kwargs).items():
				newrep = repleft * repright
				newrepweight = repleftweight * reprightweight
				if newrep in R:
					R[newrep] += newrepweight
				else:
					R[newrep] = newrepweight
		return R

class DegenerateDist(ProbabilityDist):
	def __init__(self, x=None):
		self.val = x
		assert (x is None) or (type(x).__name__ == "float")
	def mean(self, VariableDefs, QuantValues, ChangedVariables={}, **kwargs):
		return self.val
	def sqmean(self, VariableDefs, QuantValues, ChangedVariables={}):
		return self.val**2
	def var(self, ChangedVariables={}):
		return 0.
	def rvs(self, n=1, ChangedVariables={}):
		return array([self.val for i in xrange(n)])
	def __str__(self):
		return str(type(self)) + ":  " + str(self.val)
	def representatives(self, VariableDefs, QuantValues, ChangedVariables, **kwargs):
		return {self.val : 1.}

class Norm(ProbabilityDist):
	def __init__(self, m, v):
		self.m = m
		self.v = v
		#self.Dist = stats.norm(loc=m, spread=v)
	def mean(self, VariableDefs, QuantValues, ChangedVariables={}, **kwargs):
		#print "asked for mean of normal dist (m,v) =", (str(self.m),str(self.v)), "with changed variables", ChangedVariables
		return self.m.mean(VariableDefs, QuantValues, ChangedVariables, **kwargs)
	def sqmean(self, VariableDefs, QuantValues, ChangedVariables={}):
		return stats.norm(loc=self.m.mean(VariableDefs, QuantValues, ChangedVariables), spread=self.v.mean(VariableDefs, QuantValues, ChangedVariables)).moments(2)
	def var(self, VariableDefs, QuantValues, ChangedVariables={}):
		return self.v.mean(VariableDefs, QuantValues, ChangedVariables)
	def __str__(self):
		return str(type(self)) + ":  m,v=" + str((self.m, self.v))
	def representatives(self, VariableDefs, QuantValues, ChangedVariables={}):
		weights = [0.0013499, 0.0214002, 0.135905, 0.341345]
		repoffsets = [0.441771, 1.33644, 2.25553, 3.20515]
		R = {}
		for i in xrange(len(weights)):
			R[self.mean(VariableDefs, QuantValues, ChangedVariables) - repoffsets[-i-1]*self.std(VariableDefs, QuantValues, ChangedVariables)] = weights[i]
		for i in xrange(len(weights)):
			R[self.mean(VariableDefs, QuantValues, ChangedVariables) + repoffsets[i]*self.std(VariableDefs, QuantValues, ChangedVariables)] = weights[-i-1]
		return R
		# HOW THESE VALUES WERE GENERATED:
		# Probability mass from -infinity to mu-3sigma is 0.0013499
		# Half of that probability mass falls before mu-3.20515sigma, and the other half falls after that
		# So we choose mu-3.20515sigma as the representative of this segment, and give it mass 0.0013499
		# And similarly for other segments (mu-3s to mu-2s, to mu-s, to mu, to mu+s, to mu+2s, to mu+3s, to infinity)
		# How to actually find these numbers:
		#	we want integral from mu-sigma to k of pdf(x)dx to be equal to integral from k to mu of pdf(x)dx
		#	ie we want cdf(k)-cdf(mu-sigma) = cdf(mu)-cdf(k)
		#	ie cdf(k) = (cdf(mu-sigma)+cdf(mu))/2

		#TODO verify that this really actually makes sense

class Unif(ProbabilityDist):
	def __init__(self, left, right):
		self.left = left
		self.right = right
	def mean(self, VariableDefs, QuantValues, ChangedVariables={}, **kwargs):
		return (self.left.mean(VariableDefs, QuantValues, ChangedVariables, **kwargs) + self.right.mean(VariableDefs, QuantValues, ChangedVariables, **kwargs)) / 2.
	def sqmean(self, VariableDefs, QuantValues, ChangedVariables={}):
		return self.var(VariableDefs, QuantValues, ChangedVariables) - self.mean(VariableDefs, QuantValues, ChangedVariables)
	def var(self, VariableDefs, QuantValues, ChangedVariables={}):
		return (self.right.mean(VariableDefs, QuantValues, ChangedVariables, **kwargs) - self.left.mean(VariableDefs, QuantValues, ChangedVariables, **kwargs))**2 / 12.
		# variance is (right-left)^2 / 12
	def __str__(self):
		return str(type(self)) + ":  left,right=" + str((self.left, self.right))
	def representatives(self, VariableDefs, QuantValues, ChangedVariables={}, **kwargs):
		R = {}
		leftpos = self.left.mean(VariableDefs, QuantValues, ChangedVariables, **kwargs)
		rightpos = self.right.mean(VariableDefs, QuantValues, ChangedVariables, **kwargs)
		piecewidth = (rightpos - leftpos) / 5. #we divide it into 5 pieces
		for piece in xrange(1, 6):
			R[leftpos + (piece-.5)*piecewidth] = .2 #subtract .5 so we're halfway through the piece; weight is .2 because we're dealing with a fifth of the entire dist (which is weight 1)
		return R


class VariableInstance(ProbabilityDist):
	def __init__(self, name):
		self.name = name
	def GetDist(self, VariableDefs, QuantValues, ChangedVariables, **kwargs):
		if self.name in ChangedVariables:
			#print "CHANGED VARIABLE: variable", self.name, "returning:", ChangedVariables[self.name].mean(VariableDefs, QuantValues, ChangedVariables, **kwargs)
			return ChangedVariables[self.name]
		elif self.name in ("$bottomcurr", "$toppost"):
			assert self.name[1:] in kwargs, "ERROR: VariableInstance looking in kwargs for '" + str(self.name[1:]) + "' but kwargs contains only " + str(kwargs)
			#print "variable", self.name, "returning:", kwargs[self.name[1:]]
			return kwargs[self.name[1:]]
		elif self.name[0] == "Q":
			return QuantValues.val[self.name[1:]]
		else:
			#print "nothing unusual noticed - not eg in ChangedVariables, which is", ChangedVariables
			assert self.name in VariableDefs
			return VariableDefs[self.name]
	def mean(self, VariableDefs, QuantValues, ChangedVariables, **kwargs):
		#print "getting mean on variable " + str(self.name)
		#print "the distribution:", self.GetDist(VariableDefs, QuantValues, ChangedVariables, **kwargs)
		return self.GetDist(VariableDefs, QuantValues, ChangedVariables, **kwargs).mean(VariableDefs, QuantValues, ChangedVariables, **kwargs)
	#def 
	#TODO

def sqmean(D):
	"""Returns the expectation of the square of D, i.e. its second moment."""
	if "sqmean" in dir(D):
		return D.sqmean()
	elif type(D).__name__ == "rv_frozen": #if it's an rv_frozen
		return D.moment(2)
	else:
		assert False, "ERROR: trying to find sqmean of unknown type " + type(D).__name__


def MakeDistSingleWord(word, wordtype):
	assert wordtype != "operator"
	if wordtype == "bracketed_expression":
		return MakeDistribution(word[1:-1])
	elif wordtype == "number":
		return DegenerateDist(float(word))
	elif wordtype == "symbol":
		return VariableInstance(word)
	else:
		assert False, "ERROR: unknown wordtype: " + str(wordtype) + " ; word is " + str(word)

def MakeMean(args):
	"""Makes distribution that's the mean of argument distributions."""
	if len(args) == 1:
		return args[0]
	R = args[0] + args[1]
	for i in xrange(2, len(args)):
		R = R + args[i]
	R = DegenerateDist(1./len(args)) * R
	return R

def MakeNorm(args):
	if len(args) == 2:
		return Norm(args[0], args[1]) #note: the args are DISTRIBUTIONS, not constants
	else:
		assert False, "ERROR: MakeNorm hasn't been taught how to deal with <> 2 arguments"
	#TODO: let this use confidence intervals

def MakeUnif(args):
	if len(args) == 2:
		return Unif(args[0], args[1])
	else:
		assert False, "ERROR: MakeUnif hasn't been taught how to deal with <> 2 arguments"

def ExecuteFunction(func, args):
	if func == "mean":
		return MakeMean(args)
	elif func == "norm":
		return MakeNorm(args)
	elif func == "unif":
		return MakeUnif(args)
	else:
		assert False, "ERROR: unknown function " + str(func)

def MakeDistTwoWords(words, wordtypes):
	if wordtypes[0] == "operator":
		assert words[0] == "-"
		return DegenerateDist(-1) * MakeDistSingleWord(words[1], wordtypes[1])
	elif wordtypes[0] == "symbol":
		assert wordtypes[1] == "bracketed_expression"
		args = ParseArgumentSet(words[1])
		return ExecuteFunction(words[0], args)
	elif wordtypes[0] == "number" and wordtypes[1] == "symbol":
		symbolconstants = {"s":1, "m":60, "h":3600, "D":86400, "W":604800, "M":2592000, "Y":31556736, "$":1}
		assert words[1] in symbolconstants, "ERROR: unrecognized symbol-constant"
		return DegenerateDist(float(words[0]) * symbolconstants[words[1]])
	else:
		assert False, "ERROR: makedisttwowords facing unknown wordtype " + str(wordtypes[0]) + " for word " + str(words[0])
	
	#TODO: allow for the case where it's a number followed by units, eg "12d13m" or "125@"

def MakeDistMoreThanTwoWords(words, wordtypes):
	"""Split up the words at each operation; then apply the last operation to the last two words, and iterate leftwards."""
	assert len(words) > 2
	operators = []
	operatortypes = []
	terms = [[]] #list containing a list for every "block" consisting only of non-operators
	termtypes = [[]]
	curr = "term"
	for i in xrange(len(words)):
		if curr == "term":
			if wordtypes[i] in ("symbol", "number", "bracketed_expression"):
				terms[-1].append(words[i])
				termtypes[-1].append(wordtypes[i])
			else:
				curr = "operator"
				operators.append(words[i])
				operatortypes.append(wordtypes[i])
		else:
			assert wordtypes[i] != "operator", "ERROR: two operators in a row; words are " + str(words[i-1]) + " and " + str(words[i]) + " with types " + str(wordtypes[i-1]) + " and " + str(wordtypes[i])
			terms.append([])
			termtypes.append([])
			terms[-1].append(words[i])
			termtypes[-1].append(wordtypes[i])
			curr = "term"
	assert len(operators) == (len(terms) - 1)
	rightterm = TermWordsToDistribution(terms[-1], termtypes[-1])
	for ilefthand in xrange(len(terms)-2, -1, -1):
		leftterm = TermWordsToDistribution(terms[ilefthand], termtypes[ilefthand])
		rightterm = ApplyOperator(leftterm, operators[ilefthand], rightterm)
	return rightterm

def isDegenerate(d):
	return type(d).__name__ == "DegenerateDist"

def isNorm(d):
	return type(d).__name__ == "Norm"

def MakeDistSum(l, r):
	if isDegenerate(l) and isDegenerate(r):
		return DegenerateDist(l.val + r.val)
	if isNorm(l) and isNorm(r):
		return Norm(l.m + r.m, l.v + r.v)
	if isNorm(l) and isDegenerate(r):
		return Norm(l.m + DegenerateDist(r.val), l.v)
	if isNorm(r) and isDegenerate(l):
		return Norm(r.m + DegenerateDist(l.val), r.v)
	return DistSum(l, r)

def MakeDistProd(l, r):
	assert type(r).__name__ != "float", "left" + str(l) + "right" + str(r)
	if isDegenerate(l) and isDegenerate(r):
		return DegenerateDist(l.val * r.val)
	#if isNorm(l) and isDegenerate(r): #NOTE: these four lines no longer work bc we don't have a structure for exponentiation by distributions
	#	return Norm(MakeDistSum(l.m, DegenerateDist(r.val)), r.val ** l.v)
	#if isNorm(r) and isDegenerate(l):
	#	return Norm(r.m + l.val, l.val ** r.v)
	return DistProd(l, r)

def ApplyOperator(ldist, operator, rdist):
	if operator == "+":
		return ldist + rdist
	elif operator == "-":
		return ldist - rdist
	elif operator == "*":
		return ldist * rdist
	elif operator == "/":
		#return DistProd(1./ldist, rdist)
		if isDegenerate(rdist):
			return DegenerateDist(1./rdist.val) * ldist
		else:
			assert False, "ERROR: dividing by a non-degenerate distribution: " + str(operator)
		# TODO create support for dividing by other distributions
	assert False, "ERROR: operation not recognized: " + str(operator)

def TermWordsToDistribution(words, wordtypes):
	"""Given a list of words, none of which are operators, create distribution."""
	assert len(words) < 3, "ERROR: don't know how to deal with more than 2 consecutive terms"
	assert len(words) > 0
	if len(words) == 1:
		return MakeDistSingleWord(words[0], wordtypes[0])
	elif len(words) == 2:
		return MakeDistTwoWords(words, wordtypes)

def MakeDistribution(S):
	"""Takes string and creates corresponding distribution."""
	words, wordtypes = ParseIntoWords(S)
	assert len(words) > 0
	if len(words) == 1:
		return MakeDistSingleWord(words[0], wordtypes[0])
	elif len(words) == 2:
		return MakeDistTwoWords(words, wordtypes)
	else:
		return MakeDistMoreThanTwoWords(words, wordtypes)

def SplitAtTopLevel(S, c):
	"""Splits string S at top-level instances of character c; i.e., it doesn't split inside brackets."""
	R = []
	startofsection = 0
	curr = 0
	while curr < len(S):
		if S[curr] == c:
			R.append(S[startofsection:curr])
			startofsection = curr+1
		elif S[curr] == "(":
			depth = 1
			while depth:
                                curr += 1
                                assert curr < len(S), "UNCLOSED BRACKET IN EXPRESSION: " + S
                                if S[curr] == ")":
                                        depth -= 1
                                elif S[curr] == "(":
                                        depth += 1
		curr += 1
	R.append(S[startofsection:])
	return R

def ParseArgumentSet(S):
	"""Takes string of bracketed_expression as input; splits at "," operators (that are on the top level); returns list of the distributions described between the commas."""
	arglist = SplitAtTopLevel(S[1:-1], ",")
	return [MakeDistribution(arg) for arg in arglist]

def ParseIntoWords(S):
	"""Takes string as input, then splits it into numbers, operations, symbols and bracketed expressions; returns tuple of: list of words, list of what type the word is."""
	words = []
	wordtypehist = []
	char = 0
	wordletters = (
		(tuple(i for i in "0123456789."), "number"),
		(tuple(i for i in "$abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_"), "symbol"),
		(tuple(i for i in "+-*/^><="), "operator"),
		)
	while char < len(S):
		startingchar = char
		if S[char] == "(":
                        depth = 1
                        while depth:
                                char += 1
                                assert char < len(S), "UNCLOSED BRACKET IN EXPRESSION: " + S
                                if S[char] == ")":
                                        depth -= 1
                                elif S[char] == "(":
                                        depth += 1
                        char += 1
			#bracketedwords, bracketedtypes = ParseIntoWords(S[startingchar+1:char-1])
			#words.append(bracketedwords)				# if you want to recursively split up bracketed expressions, uncomment these 3 lines and comment out the 2 below
			#wordtypehist.append(bracketedtypes)
                        words.append(S[startingchar:char])	
			wordtypehist.append("bracketed_expression")
			continue
		wordtype = None
		wordletterset = None
		for letterset, typename in wordletters:
			if S[char] in letterset:
				wordtype = typename
				wordletterset = letterset
				break
		char += 1
		while (char < len(S)) and (S[char] in wordletterset):
			char += 1
		words.append(S[startingchar:char])
		wordtypehist.append(wordtype)
		assert wordtype is not None, "ERROR: wordtype seems to be None; word is " + str(S[startingchar:char])
	return words, wordtypehist



if False:
	print ParseFirstLevel("u+v+w+x+y+z")
	print MakeDist("u+v+w+x+y+z", {"u":stats.norm(loc=1.,shape=1.),"v":stats.norm(loc=3,shape=1),"w":stats.norm(loc=5,shape=1),"x":stats.norm(loc=7,shape=1),"y":stats.norm(loc=9,shape=1),"z":stats.norm(loc=11,shape=1)})
	print MakeDist("u+v+w+x+y+z", {"u":stats.norm(loc=1,shape=1),"v":stats.norm(loc=3,shape=1),"w":stats.norm(loc=5,shape=1),"x":stats.norm(loc=7,shape=1),"y":stats.norm(loc=9,shape=1),"z":stats.norm(loc=11,shape=1)}).mean()
	print MakeDist("1", {})
	
	print ; print ; print
	
	PRODDIST = MakeDist("(u*v*$double_you)/3", {"u":Norm(2,1),"v":Norm(2,1),"$double_you":Norm(2,1)})
	print PRODDIST
	print PRODDIST.mean()
	print PRODDIST.var()
	print PRODDIST.std()
	print PRODDIST.rvs(10)
if False:
	print ParseIntoWords("u+v+x+(y+z)")
	D = MakeDistribution("mean(norm(1,1),norm(1,1))")
	print D
	print D.mean()
	print D.var()

if False:
	N = MakeDistribution("norm(13,17)")
	weightedsum = 0.
	weightedsqdev = 0.
	for rep in N.representatives():
		print str(rep[0]) + "\t" + str(rep[1])
		weightedsum += rep[0] * rep[1]
		weightedsqdev += (rep[0] - 13.)**2 * rep[1]
	print "weightedsum", weightedsum
	print "weightedsqdev", weightedsqdev

if False:
	THEDIST = MakeDistribution("((norm(13,3)*2)+norm(100,0.5))+10")
	print THEDIST.representatives()
	replist = []
	rwlist = []
	for rep, rw in THEDIST.representatives().items():
		print str(rep) + "\t" + str(rw)
		replist.append(rep)
		rwlist.append(rw)
	print
	replist.sort()
	rwlist.sort()
	for rep in replist:
		print rep
	print
	for rw in rwlist:
		print rw

if False:
	THEDIST = MakeDistribution("$bottomcurr+1")
	print THEDIST.mean(bottomcurr=1.)
