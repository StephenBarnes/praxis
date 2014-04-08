#!/usr/bin/env python

import os, sys
from copy import copy, deepcopy
from math import log
from expressions import *


DISCOUNTING = 0.8 ** (1./(60*60*24*365.25)) #i.e., I'm indifferent between 100U now and 80U a year from now

VOIHORIZON = 60*60*24

AgosNames = set()
QuantityNames = set()
VariableNames = set()
AgosDefinitions = {}
QuantityDefinitions = {}
VariableDefinitions = {}
QuantityDependencies = {}

class Configuration():
	def __init__(self):
		self.ExecutableAgosi = set() #agosi that can be executed in self
		self.DepletedAgosi = set() #agosi that have already been executed
		self.PrereqqedAgosi = set() #agosi that have unsatisfied prereqs
		self.ChangedVariables = {} #variables that have different values here than they do in current reality

		self.QuantityValues = QuantityValueSet() #the amounts of various quantities that I have in configuration self

		self.NewlyExecutableAgosi = set() #the set of agosi that are executable here but weren't executable in whatever created this configuration (e.g. an ApplyAgos)

		self.CreatableValueMemo = {}
		self.AgosIndirectValueMemo = {}

		self.QuantityValues = QuantityValueSet()
	
	#def InitializeAgosiSets(self, Agosi):
	#	"""Checks prereqs, repeatability, and whether executed yet, for all agosi named in Agosi, then puts them into the correct agos sets (Executable-, Depleted-, Prereqqed-).
	#	This is for right after creating the main Configuration and reading the agos definitions from the configuration file."""
	#	for agosname in Agosi:
	#		self.ExecutableAgosi.add(agosname)
	#	#TODO put some things in prereqqedagosi
	
	def MakeAgosiExecutable(self):
		"""Check agosi in PrereqqedAgosi and see if their prereqs are satisfied; if so, move them over to ExecutableAgosi."""
		self.NewlyExecutableAgosi = set()
		for PA in self.PrereqqedAgosi:
			if AgosDefinitions[PA].PrereqsSatisfied(self.QuantityValues, self.DepletedAgosi):
				#self.PrereqqedAgosi.remove(PA)
				self.ExecutableAgosi.add(PA)
				self.NewlyExecutableAgosi.add(PA)
		for PA in self.NewlyExecutableAgosi:
			self.PrereqqedAgosi.remove(PA)

	def __str__(self):
		R = ""
		R += "\nExecutable Agosi: " + str(self.ExecutableAgosi)
		R += "\nDepleted Agosi: " + str(self.DepletedAgosi)
		R += "\nPrereqqed Agosi: " + str(self.PrereqqedAgosi)
		R += "\nChanged Variables: " + str(self.ChangedVariables)
		R += "\nQuantity Values: " + str(self.QuantityValues)
		return R
	def Rustle(self):
		self.QuantityValues.Rustle()
		self.CreatableValueMemo = {}
		self.AgosIndirectValueMemo = {}
	def GetQuantityValue(self):
		return self.QuantityValues.GetQuantityValue(self.ChangedVariables)
	def ApplyAgos(self, agosname):
		#assert agosname in self.ExecutableAgosi, "ERROR: trying to apply non-executable agos " + str(agosname)
		R = deepcopy(self)
		for QE in AgosDefinitions[agosname].QuantityEffects:
			R.QuantityValues.ApplyQuantityStraightEffect(QE, self.ChangedVariables)
		if not AgosDefinitions[agosname].repeatable:
			R.ExecutableAgosi.remove(agosname)
			R.DepletedAgosi.add(agosname)
		R.MakeAgosiExecutable()
		# TODO think about how to get the "cause" of the value discrepancy, by examining the trickledvalues of quantities in the before and after configurations (respectively self, NewConfiguration)
		R.Rustle() #we've just fiddled with NewConfiguration, so we must redo our calculations about it
		return R
	#def GetAgosValueRate(self, agosname):
	#	return self.GetAgosValue(agosname) / AgosDefinitions[agosname].ImmediateTimeReq
	def AgosDirectValue(self, agosname):
		#assert agosname in self.ExecutableAgosi, "ERROR: trying to get (direct) value of non-executable agos " + str(agosname)
		ConfigPostAgos = self.ApplyAgos(agosname)
		return ConfigPostAgos.GetQuantityValue() - self.GetQuantityValue()
	def AgosAnnuityValue(self, agosname):
		R = 0.
		for A in AgosDefinitions[agosname].QuantityAnnuityEffects:
			R += A.Value([], VariableDefinitions, self.QuantityValues, self.ChangedVariables)
		return R
	def AgosIndirectValue(self, agosname):
		"""Returns the greatest utility rate that we can get by doing action agosname and then doing new possibilities it opens up (or not doing any new possibilities, if that's better). I.e., it can return the exact same thing as AgosDirectValue if that's greater than anything we get through appending further later actions."""
		#assert agosname in self.ExecutableAgosi, "ERROR: trying to get (indirect) value of non-executable agos " + str(agosname)
		#if agosname in self.AgosIndirectValueMemo: #NOTE: not memoizing any more bc we're changing variables
			#print "agosindirectvalue returns VIA MEMO:", self.AgosIndirectValueMemo[agosname]
			#return self.AgosIndirectValueMemo[agosname]
		#print; print "AgosIndirectValue for agos", agosname
		ConfigPostAgos = self.ApplyAgos(agosname)
		AnnuityVal = self.AgosAnnuityValue(agosname)
		assert type(AgosDefinitions[agosname].ImmediateTimeReq).__name__ != "float"
		MaxU, TotalTimeReq, BestFutureAgosi = ConfigPostAgos.ContinueAgosChain(list(ConfigPostAgos.NewlyExecutableAgosi), DegenerateDist(self.GetQuantityValue()), AgosDefinitions[agosname].ImmediateTimeReq, AnnuityVal) 
		#TODO: GetQuantityValue should return a distribution, so I should be able to pass that distribution into ContinueAgosChain! unlike above, where I have to make it into a degeneratedist and then pass it in
		assert TotalTimeReq.mean(VariableDefinitions, self.QuantityValues, self.ChangedVariables) != 0, "ERROR: totaltimereq seems to be zero, in function AgosIndirect value given agos:" + agosname
		#self.AgosIndirectValueMemo[agosname] = (MaxU/TotalTimeReq.mean(VariableDefinitions, self.QuantityValues, self.ChangedVariables), MaxU, TotalTimeReq.mean(VariableDefinitions, self.QuantityValues, self.ChangedVariables), [agosname] + BestFutureAgosi) #commented out bc we're not memoizing anymore
		#print "agosindirectvalue returns:", (MaxU/TotalTimeReq.mean(VariableDefinitions, self.QuantityValues, self.ChangedVariables), MaxU, TotalTimeReq.mean(VariableDefinitions, self.QuantityValues, self.ChangedVariables), [agosname] + BestFutureAgosi)
		return MaxU/TotalTimeReq.mean(VariableDefinitions, self.QuantityValues, self.ChangedVariables), MaxU, TotalTimeReq.mean(VariableDefinitions, self.QuantityValues, self.ChangedVariables), [agosname] + BestFutureAgosi
		# ie, it returns: (U_per_second, totalu, seconds, agosseries)
		# TODO change this thing so it returns a dictionary with meaningful keys instead of just a list
		#return MaxU, [agosname] + BestFutureAgosi
	def ContinueAgosChain(self, NewAgosi, StartingQuantityValue, TimeIntoFuture, AccumulatedAnnuityValue):
		MaxUSoFar = self.GetQuantityValue() + AccumulatedAnnuityValue - StartingQuantityValue.mean(VariableDefinitions, self.QuantityValues, self.ChangedVariables)
		TotalTimeReqSoFar = TimeIntoFuture
		BestFutureAgosi = []
		BestCurrentAgos = None
		for i in xrange(len(NewAgosi)):
			NewConfig = self.ApplyAgos(NewAgosi[i])
			AnnuityVal = NewConfig.AgosAnnuityValue(NewAgosi[i])
			newnewagosi = list(NewConfig.NewlyExecutableAgosi)
			currU, currTimeReq, currAgosi = NewConfig.ContinueAgosChain(NewAgosi[i+1:] + newnewagosi, StartingQuantityValue, TimeIntoFuture + AgosDefinitions[NewAgosi[i]].ImmediateTimeReq, AccumulatedAnnuityValue + AnnuityVal)
			if ((currU+AnnuityVal)/(currTimeReq).mean(VariableDefinitions, self.QuantityValues, self.ChangedVariables)) > (MaxUSoFar/TimeIntoFuture.mean(VariableDefinitions, self.QuantityValues, self.ChangedVariables)):
				MaxUSoFar = currU + AnnuityVal
				TotalTimeReqSoFar = currTimeReq
				BestFutureAgosi = currAgosi
				BestCurrentAgos = NewAgosi[i]
		if BestCurrentAgos is None:
			return MaxUSoFar, TotalTimeReqSoFar, [] #because best option is to take no further action
		else:
			return MaxUSoFar, TotalTimeReqSoFar, [BestCurrentAgos] + BestFutureAgosi
	def ValueCreatedBy(self, AgosSeries):
		CurrValue = self.GetQuantityValue()
		NewestConfig = self
		for Agos in AgosSeries:
			NewestConfig = NewestConfig.ApplyAgos(Agos)
		NewValue = NewestConfig.GetQuantityValue()
		return NewValue - CurrValue
	def ApplyEvidence(self, VarName, NewDist):
		R = deepcopy(self)
		R.MakeAgosiExecutable()
		R.ChangedVariables[VarName] = NewDist
		R.Rustle() #we've just fiddled with NewConfiguration, so we must redo our calculations about it
		return R
	def CreatableValue(self, time=1.):
		#if (time, self.ChangedVariables) in self.CreatableValueMemo:
		#	return self.CreatableValueMemo[(time, self.ChangedVariables)] #NOTE: not hashing bc self.ChangedVariables is a dict and you can't hash a dict
		ValueCreated = 0.
		MaximizingAgosSeries = []

		AllAgosiSorted = [(list(self.AgosIndirectValue(agosname)) + [agosname]) for agosname in self.ExecutableAgosi]
		# remember, agosindirectvalue returns: (U_per_second, totalu, seconds, agosseries)
		AllAgosiSorted.sort(reverse=True) #sort so that the highest-value thing comes first
		#print "allagosisorted", AllAgosiSorted

		UsedAgosi = set()

		icurr = 0
		while time > 0:
			if icurr >= len(AllAgosiSorted):
				break
			NextAgosDetails = AllAgosiSorted[icurr]
			#print "nextagosdetails:", NextAgosDetails
			NextAgosTimeReq = NextAgosDetails[2]
			EverythingUnused = True
			for agosname in NextAgosDetails[3]:
				if agosname in UsedAgosi:
					if not AgosDefinitions[agosname].repeatable:
						EverythingUnused = False
				UsedAgosi.add(agosname)
			if not EverythingUnused:
				icurr += 1
				continue
			MaximizingAgosSeries.append((AllAgosiSorted[icurr][0], AllAgosiSorted[icurr][4]))
			if NextAgosTimeReq > time:
				ValueCreated += (time / NextAgosTimeReq) * NextAgosDetails[1]
				time = 0.
			else:
				ValueCreated += NextAgosDetails[1]
				time -= NextAgosTimeReq
				#if not AgosDefinitions[NextAgosDetails[2]].repeatable:
				#	icurr += 1
				# now we look for repeatable subtasks that weren't possible as top-level actions, and add those to the list of actions
				for additionalagosname in NextAgosDetails[3]:
					#print "additionalagos", additionalagosname
					if AgosDefinitions[additionalagosname].repeatable:
						toadd = list(self.AgosIndirectValue(additionalagosname)) + [additionalagosname]
						if toadd not in AllAgosiSorted:
							AllAgosiSorted.append(toadd)
						#print "appended; allagosi is now", AllAgosiSorted
				AllAgosiSorted.sort(reverse=True)
				icurr = 0

		#TODO: the algorithm above could probably be sped up somehow; I'm not even sure it's correct. Think on it more.

		#self.CreatableValueMemo[(time,self.ChangedVariables)] = (ValueCreated, MaximizingAgosSeries) #commented out bc we're not doing memoization any more
		return ValueCreated, MaximizingAgosSeries
	
	def EvidenceValue(self, varname, newdist):
		# first, get optimal agos-sequence in current world
		OldOptimalAgosSequence = self.CreatableValue(VOIHORIZON)[1]
		# then get the value of that sequence in the new world
		PostEvidenceWorld = self.ApplyEvidence(varname, newdist)
		ValueOfOldAgosSeqInNewWorld = PostEvidenceWorld.ValueCreatedBy([i[1] for i in OldOptimalAgosSequence]) #TODO fix this so it takes into account differences in time requirements of agosi etc.
		# then get optimal agos-sequence in the new world and the value of that sequence in the new world
		ValueOfNewAgosSeqInNewWorld, NewOptimalAgosSequence = PostEvidenceWorld.CreatableValue(VOIHORIZON)
		
		#print
		#print "changed variable is", varname, "which is now", newdist
		#print "old optimal agosseq:", OldOptimalAgosSequence
		#print "new optimal agosseq:", NewOptimalAgosSequence
		#print
		
		# return (new world value of new optimal agos seq) - (new world value of old optimal agos sequence)
		return ValueOfNewAgosSeqInNewWorld - ValueOfOldAgosSeqInNewWorld
	
	def VariableEVOPI(self, varname):
		assert varname not in self.ChangedVariables.keys()
		R = 0.
		for rep in VariableDefinitions[varname].representatives(VariableDefinitions, self.QuantityValues, self.ChangedVariables).items():
			R += self.EvidenceValue(varname, DegenerateDist(rep[0])) * rep[1]
		return R

def MakeAgosQuantEffectFunction(words):
	assert len(words) == 1
	text = words[0]
	if text.find("$") == -1:
		if text[0] == "-":
			text = "$bottomcurr" + text
		else:
			text = "$bottomcurr+" + text
	return MakeDistribution(text)

def MakeQuantQuantEffectFunction(words):
	assert len(words) == 1
	text = words[0]
	if text.find("$") == -1:
		text = "$bottomcurr+$toppost*(" + text + ")"
	#TODO more complex rules, like below
	return MakeDistribution(text)

def CreateFunction(text, context):
	"""Creates a function from the text; functions are used in defining the effects of quantities and agosi. Context can be "quantity-quanteffect", "agos-quanteffect"."""
	assert context in ["quantity-quanteffect", "agos-quanteffect"], "ERROR: function CreateFunction, unrecognized context \"" + str(context) + "\" in function \"" + str(text) + "\""
	if context == "quantity-quanteffect":
		if text.find("$") == -1:
			text = "$bn+$tn*" + text
		elif text.find("$b") == -1: #used to call eg $bn "$bottomnew"; after changing it, spent like 45 minutes looking for a bug that turned out to be caused by this line still being "$bottom" instead of "$b" - it would make the effect into things like "$bn+$bn+$bn+...+$bn+$tn*1"
			text = "$bn+" + text
		elif text[0] == "+":
			text = "$bn" + text
	simplesubstitutions = ["tn", "bo", "bn"]
	#TODO put complex substitutions here too, ie substitutions that aren't just $i->i, eg for computing the number of seconds left in my (pre-singularity) life (which is a probability distribution)
	for simplesub in simplesubstitutions:
		text = text.replace("$" + simplesub, simplesub)
	arglist = ", ".join(simplesubstitutions)
	return eval("lambda " + arglist + ": " + text)
	# FOR QUANTITY-QUANTEFFECTS:
				# tn = topnew, ie the value of the top after the trickling
				# bn = bottomnew, ie the value of the bottom after the trickling done SO FAR
				# bo = bottom-original, ie the value of the bottom before all trickling
				#SUMMARY OF THE CURRENT RULES FOR PARSING THESE THINGS:
				# if you use no variables, it's assumed you mean each unit of the top becomes <effect> units of the bottom
				# if you use variables but completely ignore the existing and updated values of the bottom thing, we assume you mean that <effect> is to be ADDED to the current bottom value
				# if you start with a +, it's assumed to be shorthand for "$bn+"

def TopoSort(Names, Dependencies):
	"""Does topological sort on Names according to the Dependencies specified; returns elements of Names in correct order."""
	R = []
	unprocessed = set((i for i in Names))
	processed = set()
	unsatisfieddependencycount = {} #we use this because we don't want to modify Dependencies, and this is quicker than copying the Dependencies dict
	for quantity in QuantityNames:
		unsatisfieddependencycount[quantity] = len(Dependencies[quantity])
	while unprocessed:
		currquantity = None
		for poscurrquantity in unprocessed:
			if unsatisfieddependencycount[poscurrquantity] == 0:
				currquantity = poscurrquantity
				break
		assert currquantity is not None, "ERROR: loop in dependency graph prevents topological sort; looping elements seem to be " + str(unprocessed)
		unprocessed.remove(currquantity)
		processed.add(currquantity)
		R.append(currquantity)
		for name in Names:
			if currquantity in Dependencies[name]:
				unsatisfieddependencycount[name] -= 1
	#print "toposort returned: ", R
	return R

def RemoveCountPrefixes(line, char):
	if line == "":
		return ("", 0)
	R = 0
	while line[R] == char:
		R += 1
		if R >= len(line):
			return ("", 0)
	return (line[R:], R) if R else (line, 0)

def ParseTabLevels(S, isfile=True):
	"""Reads the lines in string <S> (or, if <isfile>, reads lines from file called <S>) and then returns list containing its lines (with line numbers) with nesting according to indentation."""
	infile = None
	Lines = None
	if isfile:
		infile = open(S, "r")
	else:
		Lines = S.split("\n")
	R = []
	linenum = 0
	nestlevel = -1 #set to -1 so first line must start with 0 tabs
	while True:
		line = None
		if isfile:
			line = infile.readline()
			if line == "":
				break #readline() returned blank, so it's EOF
			line = line[:-1] #(removing final newline)
		else:
			if linenum >= len(Lines):
				break #at the end of the lines, so it's EOF
			line = Lines[linenum]
		linenum += 1
		line, numtabs = RemoveCountPrefixes(line, "\t")
		if line == "":
			continue #line contains only tabs and/or trailing newline; ignore it
		if line[0] == "#":
			continue #line is a comment, so ignore it
		assert numtabs <= (nestlevel + 1), "ERROR: too many tabs on line " + str(linenum) + ": " + line
		placetoadd = R
		for i in xrange(numtabs):
			placetoadd = placetoadd[-1]
		placetoadd.append([(linenum, line)])
		nestlevel = numtabs
	return R

def TreeFormat(L):
	"""Transforms a nested list into a tree-format string."""
	def TreeFormatR(L):
		if L is None:
			return ""
		if len(L) == 0:
			return ""
		R = "\n"
		R += str(L[0][0]) + "|" + str(L[0][1])
		for l in L[1:]:
			R += TreeFormatR(l).replace("\n", "\n\t")
		return R
	retval = ""
	for toplevelline in L:
		retval += TreeFormatR(toplevelline)
	return retval

def RustleGlobalVariables():
	global AgosNames
	global QuantityNames
	global VariableNames
	global AgosDefinitions
	global QuantityDefinitions
	global VariableDefinitions
	global QuantityDependencies

	AgosNames = set()
	QuantityNames = set()
	VariableNames = set()
	AgosDefinitions = {}
	QuantityDefinitions = {}
	VariableDefinitions = {}
	QuantityDependencies = {}

def ParseConfig(S, isfile=False):
	"""Given a configuration file or string, creates definitions (variables, agosi, quantities) and returns a configuration for current worldstate."""
	RustleGlobalVariables()
	R = Configuration()
	LineList = ParseTabLevels(S, isfile)
	ParseConfigGivenLevelParsedLines(LineList, R)
	for agosname in AgosNames:
		R.PrereqqedAgosi.add(agosname)
	R.MakeAgosiExecutable()
	return R

def ParseConfigGivenLevelParsedLines(LineList, CurrConfig):
	"""Parses the lines of Lines (which is assumed to be the output of ParseTabLevels()), and incorporates them into CurrConfig."""
	def GetNames(LL):
		"""Given LineList, gets only the names of the agosi and quatities. NB: this doesn't get the names of the variables."""
		for toplevel in LL:
			if toplevel[0][1][0] == "A":
				AgosNames.add(toplevel[0][1].split()[1])
			elif toplevel[0][1][0] == "Q":
				quantityname = toplevel[0][1].split()[1]
				QuantityNames.add(quantityname)
				QuantityDependencies[quantityname] = set()
	GetNames(LineList)
	for toplevel in LineList:
		#toplevel[0] is first line, toplevel[0][1] is text of first line, toplevel[0][1][0] is first char of text of first line
		assert toplevel[0][1][0] != "#", "ERROR: comment somehow escaped ParseTabLevels: " + str(toplevel)
		if toplevel[0][1][0] == "A":
			ParseAgosDefinition(toplevel, CurrConfig)
		elif toplevel[0][1][0] == "Q":
			ParseQuantityDefinition(toplevel, CurrConfig)
		elif toplevel[0][1][0] == "V":
			ParseVariableDefinition(toplevel, CurrConfig)
		else:
			assert False, "ERROR: ParseConfigGivenLevelParsedLines doesn't know how to process line #" + str(toplevel[0][0]) + ", which is: " + str(toplevel[0][1])

def ParseAgosDefinition(toplevel, CurrConfig):
	toplevelwords = toplevel[0][1].split()
	assert len(toplevelwords) == 2, "ERROR: too many words on first line of agos definition; line # " + str(toplevel[0][0]) + ", which is " + str(toplevel[0][1])

	NewAgos = AgosDefinition(toplevel[0][0], toplevelwords[1])
	AgosDefinitions[toplevelwords[1]] = NewAgos
	
	for subline in toplevel[1:]:
		words = subline[0][1].split()
		linenum = subline[0][0]
		if words[0] == "agos_prereq":
			assert len(words) == 2, "ERROR: too many words on quant_prereq line; line # " + str(linenum)
			assert len(subline) == 1, "ERROR: quant_prereq can't have sublines; line # " + str(linenum) + ", which is " + str(subline[0][1])
			NewAgos.AgosPrereqs.add(words[1])
		elif words[0] == "quant_prereq":
			assert len(words) == 4, "ERROR: quant_prereq must have exactly 4 words; line # " + str(linenum)
			assert len(subline) == 1, "ERROR: quant_prereq can't have sublines; line # " + str(linenum) + ", which is " + str(subline[0][1])
			NewAgos.QuantityPrereqs.add(QuantityPrereq(linenum, *words[1:]))
		elif words[0] == "repeatable":
			NewAgos.repeatable = True
		elif words[0][0] == "@":
			NewAgos.QuantityAnnuityEffects.add(QuantityAnnuity(linenum, words))
		elif words[0] == "now":
			NewAgos.ImmediateTimeReq = NewAgos.ImmediateTimeReq + MakeDistribution(words[1])
		elif words[0] in QuantityNames:
			NewAgos.QuantityEffects.add(QuantityEffect(linenum, words[0], MakeAgosQuantEffectFunction(words[1:])))
		else:
			assert False, "ERROR: ParseAgosDefinition doesn't know what to do with line # " + str(linenum) + ", which is " + str(subline[0][1])
	
	assert NewAgos.ImmediateTimeReq > 0, "ERROR: agos definition implies ImmediateTimeReq is 0" #NOTE: should remove this, but fix the parts where we divide by elapsed time

def ParseQuantityDefinition(toplevel, CurrConfig):
	toplevelwords = toplevel[0][1].split()
	assert len(toplevelwords) in (2,3), "ERROR: quantity definition lines must have 2 or 3 words; line # " + str(toplevel[0][0]) + ", which is " + str(toplevel[0][1])

	NewQuantity = QuantityDefinition(toplevel[0][0], toplevelwords[1])
	QuantityDefinitions[toplevelwords[1]] = NewQuantity
	if len(toplevelwords) == 2:
		CurrConfig.QuantityValues.val[toplevelwords[1]] = DegenerateDist(0.)
	else:
		CurrConfig.QuantityValues.val[toplevelwords[1]] = MakeDistribution(toplevelwords[2])
	
		
	for subline in toplevel[1:]:
		words = subline[0][1].split()
		linenum = subline[0][0]
		if words[0][0] == "@":
			#QuantityDependencies[toplevelwords[1]].add(words[0][1:])
			QuantityDependencies[words[0][1:]].add(toplevelwords[1])
			NewQuantity.QuantityAnnuityEffects.add(QuantityAnnuity(linenum, words, NewQuantity))
		elif words[0] in QuantityNames:
			#QuantityDependencies[toplevelwords[1]].add(words[0])
			QuantityDependencies[words[0]].add(toplevelwords[1])
			NewQuantity.QuantityEffects.add(QuantityEffect(linenum, words[0], MakeQuantQuantEffectFunction(words[1:])))
		else:
			assert False, "ERROR: ParseQuantityDefinition doesn't know what to do with line # " + str(linenum) + ", which is " + str(subline[0][1])
	

def ParseVariableDefinition(toplevel, CurrConfig):
	toplevelwords = toplevel[0][1].split()[1:] #remove the first "V"
	#TODO: let definitions be more than one line, eg allow recursive "weighted aggregation" structures
	VariableDefinitions[toplevelwords[0]] = MakeDistribution(toplevelwords[1])
	VariableNames.add(toplevelwords[0])

class Statement(object):
	"""Parent class to everything that has a linenumber: definitions, effects, etc."""
	def __init__(self, linenum):
		self.linenum = linenum

class AgosDefinition(Statement):
	def __init__(self, linenum, agosname, description=""):
		self.linenum = linenum
		self.agosname = agosname
		self.description = description
		self.repeatable = False
		self.QuantityEffects = set()
		self.QuantityPrereqs = set()
		self.QuantityAnnuityEffects = set()
		self.AgosPrereqs = set()
		self.ImmediateTimeReq = DegenerateDist(0.)
	def PrereqsSatisfied(self, QuantValues, DepletedAgosi):
		return self.QuantityPrereqsSatisfied(QuantValues) and self.AgosPrereqsSatisfied(DepletedAgosi)
	def QuantityPrereqsSatisfied(self, QuantValues):
		for QuantPrereq in self.QuantityPrereqs:
			pass #TODO
		return True
	def AgosPrereqsSatisfied(self, DepletedAgosi):
		for AP in self.AgosPrereqs:
			if AP not in DepletedAgosi:
				return False
		return True

class QuantityDefinition(Statement):
	def __init__(self, linenum, quantname, description=""):
		self.linenum = linenum
		self.quantname = quantname
		self.description = description
		self.QuantityEffects = set()
		self.QuantityAnnuityEffects = set()

class QuantityPrereq(Statement):
	def __init__(self, linenum, left, comparator, right):
		self.linenum = linenum
		self.left = left
		self.right = right
		self.comparator = comparator #the < or > or = that's used to compare left and right
		self.variables = set() #names of all variables that appear in either term (so we can more efficiently check whether truth-value has changed)
			#TODO infer this on creation
	def Satisfied(self, QuantValues, ChangedVariables):
		pass #TODO

class QuantityEffect(Statement):
	def __init__(self, linenum, quantname, effect):
		self.linenum = linenum
		self.quantname = quantname
		self.effect = effect
	def __str__(self):
		return "QuantityEffect, linenum " + str(self.linenum) + ", quantname " + str(self.quantname) + ", effect " + str(self.effect)

class QuantityAnnuity(Statement):
	def __init__(self, linenum, words, parentquant=None): #parentquant is the quantity to which the annuity belongs, OR we make it None if the annuity is from an agos
		self.linenum = linenum
		self.parentquant = parentquant
		self.quantname = words[0][1:]
		if self.parentquant is None:
			self.effect = QuantityEffect(linenum, self.quantname, MakeAgosQuantEffectFunction([words[1]])) #using the function intended for agosi bc annuity is kinda like an agos's effect, eg it doesn't have a coefficient like a quantity has a quantity-amount; #note that we have to send words[1] as a list of length 1 or it won't work
		else:
			assert words[1][0] not in ("/", "%", "r", "l"), "ERROR: you seem to have forgotten to give the actual effect of an annuity; line # " + str(self.linenum) + ", which is " + str(words)
			self.effect = QuantityEffect(linenum, self.quantname, MakeQuantQuantEffectFunction([words[1]])) #using the function intended for agosi bc annuity is kinda like an agos's effect, eg it doesn't have a coefficient like a quantity has a quantity-amount; #note that we have to send words[1] as a list of length 1 or it won't work
		self.repinterval = DegenerateDist(60*60*24.) #assume that the interval of repetition is one day
		self.decay = DegenerateDist(1.)
		self.numreps = None
		self.duration = None #only used if this gets set and numreps isn't set
		for w in words[2:]: #we process all the words that aren't just straightforward quantname or effect
			if w[0] == "/":
				self.repinterval = MakeDistribution(w[1:])
			elif w[0] == "l":
				if w[1] == "A": #lasts for the rest of my life
					self.duration = DegenerateDist(60*60*24*365.24*(80-20)) #number of seconds left in life
				else:
					self.duration = MakeDistribution(w[1:]) 
			elif w[0] == "r":
				self.numreps = MakeDistribution(w[1:])
			elif w[0] == "%":
				self.decay = MakeDistribution(w[1:]) * DegenerateDist(1./100) #NOTE: this is the decay PER INTERVAL
			else:
				assert False, "ERROR: unknown prefix symbol in annuity definition"
		if self.duration is not None:
			assert self.numreps is None, "ERROR: both duration and numreps defined for annuity"
		else:
			assert self.numreps is not None, "ERROR: neither duration nor numreps defined for annuity"
	def Value(self, annuitiestoremove, VariableDefs, QuantValues, ChangedVariables, **kwargs):
		interval = self.repinterval.mean(VariableDefs, QuantValues, ChangedVariables, **kwargs)
		reps = 0.
		if self.numreps is None:
			if self.duration is None:
				reps = (60*60*24*365.24*(80-20.)) / interval #assume it lasts for rest of life
			else:
				reps = self.duration.mean(VariableDefs, QuantValues, ChangedVariables, **kwargs) / interval
		else:
			assert self.duration is None
			reps = self.numreps.mean(VariableDefs, QuantValues, ChangedVariables)
		decay = self.decay.mean(VariableDefs, QuantValues, ChangedVariables)
		#simpleeffect = self.effect.mean(VariableDefs, QuantValues, ChangedVariables={}, **kwargs)
		# we approximate the value each rep as what the value would be right now (in terms only of quantity value, not eg the satisfaction of prereqs)
		PostEffect = deepcopy(QuantValues)
		if self.parentquant is None:
			PostEffect.ApplyQuantityStraightEffect(self.effect, ChangedVariables)
		else:
			#PostEffect.ApplyQuantityTricklingEffect(self.effect, self.parentquant.quantname)
			#print "names:", self.quantname, self.parentquant.quantname
			PostEffect.val[self.quantname] = DegenerateDist(self.effect.effect.mean(VariableDefinitions, QuantValues, ChangedVariables, bottomcurr=QuantValues.val[self.quantname], toppost=QuantValues.val[self.parentquant.quantname]))
			#FIXME: we shouldn't be "flattening" quantities' distributions into degenerate distributions like this, or should we?
			#newamt = quantnumtransform.mean(VariableDefinitions, self, bottomcurr=DegenerateDist(self.quantitytrickledvalues[Effect.quantname]), toppost=DegenerateDist(self.quantitytrickledvalues[Originator]))
			#newamt = quantnumtransform.mean(VariableDefinitions, self, bottomcurr=self.val[Effect.quantname])
		value = PostEffect.GetQuantityValueWithAnnuityRemoved([self]+annuitiestoremove, ChangedVariables) - QuantValues.GetQuantityValueWithAnnuityRemoved([self]+annuitiestoremove, ChangedVariables)
		#print "VALUE ", value

		#bottomcurr=DegenerateDist(trickledwithoutannuity[effect.quantname]), toppost=DegenerateDist(trickledwithoutannuity[name])
		
		totaldecayperinterval = 1. / (decay * (DISCOUNTING**interval)) #this is the decay per interval when we also take into account time-discounting
		#print "decay", decay
		#print "discounting", DISCOUNTING
		#print "interval", interval
		#print "totaldecayperinterval", totaldecayperinterval
		#print "reps", reps
		
		#print "ANNUITY VALUE ", value * ( (1 - (totaldecayperinterval)**(-reps)) / (totaldecayperinterval-1) )
	
		return value * ( (1 - (totaldecayperinterval)**(-reps)) / (totaldecayperinterval-1) ) #this is the usual formula for the present value of an annuity
		



class QuantityValueSet(Statement):
	def __init__(self):
		self.val = {} #stores the quantity values, ie maps from quantity names to the DISTRIBUTIONS over them
		self.Rustle()
	def Rustle(self):
		self.quantitytrickledvalues = {} #stores the amounts of each quantity after we've taken into account the splits of quantities into other quantities
		for qname in QuantityNames:
			self.quantitytrickledvalues[qname] = 0.
		self.approxcontributions = {} #stores the amount in quantitytrickledvalues that was contributed by every quantity; it's a dictionary, with keys the names of quantities and values dictionaries (from names of quantities to their contributions); it's approximate because not all quantities simply add/subtract sth, they can arbitrarily change it, so we just take their EFFECT on it, and also the order of evaluation of quantities could change things
		for qname in QuantityNames:
			self.approxcontributions[qname] = {}
		self.quantitiesshifted = False
	def GetQuantityValue(self, ChangedVariables):
		if not self.quantitiesshifted:
			self.ShiftQuantityValues(ChangedVariables)
		#self.ShiftQuantityValues(ChangedVariables)
		return self.quantitytrickledvalues["direct_value"]
	def GetQuantityValueWithAnnuityRemoved(self, annuitiestoremove, ChangedVariables):
		trickledwithoutannuity = self.ShiftQuantityValuesWithAnnuityRemoved(annuitiestoremove, ChangedVariables)
		return trickledwithoutannuity["direct_value"]
	def ShiftQuantityValuesWithAnnuityRemoved(self, annuitiestoremove, ChangedVariables):
		trickledwithoutannuity = {}
		for quantname in QuantityNames:
			#print self.val
			#print self.val[quantname]
			trickledwithoutannuity[quantname] = self.val[quantname].mean(VariableDefinitions, self)
			assert type(self.val[quantname]).__name__ != "float"
		#print trickledwithoutannuity
		trickleorder = TopoSort(QuantityNames, QuantityDependencies)
		for currtrickler in trickleorder:
			self.TrickleQuantityWithAnnuityRemoved(annuitiestoremove, trickledwithoutannuity, currtrickler, ChangedVariables)
		return trickledwithoutannuity
	def TrickleQuantityWithAnnuityRemoved(self, annuitiestoremove, trickledwithoutannuity, name, ChangedVariables):
		for effect in QuantityDefinitions[name].QuantityEffects:
			quantnumtransform = effect.effect
			newamt = quantnumtransform.mean(VariableDefinitions, self, ChangedVariables, bottomcurr=DegenerateDist(trickledwithoutannuity[effect.quantname]), toppost=DegenerateDist(trickledwithoutannuity[name]))
			trickledwithoutannuity[effect.quantname] = newamt
		for annuity in QuantityDefinitions[name].QuantityAnnuityEffects:
			if annuity in annuitiestoremove:
				continue
			trickledwithoutannuity["direct_value"] += annuity.Value(annuitiestoremove, VariableDefinitions, self, ChangedVariables)
			# note: remember what annuity.Value actually DOES! It does not give the value of a single instance of the annuity. It does not give some sort of "time-flattened to present" value that we can just add to quantitytrickledvalues[annuity.quantname]. Rather, it returns the ULTIMATE VALUE of the annuity, ie how much the annuity contributes to direct_value. For determining that, it looks at the value of a single annuity-payout now and assumes every future annuity-payout will have the same payout. We DO NOT have to multiply annuity.Value by quantitytrickledvalues[name].
	def ShiftQuantityValues(self, ChangedVariables):
		for quantname in QuantityNames:
			self.quantitytrickledvalues[quantname] = self.val[quantname].mean(VariableDefinitions, self)
			assert type(self.val[quantname]).__name__ != "float"
			self.approxcontributions[quantname] = {"STARTING-VALUE":self.quantitytrickledvalues[quantname]}
		#print "|SHIFTING QUANTITY VALUES"
		#print "|after initialization:", self.quantitytrickledvalues
		trickleorder = TopoSort(QuantityNames, QuantityDependencies)
		#print "|trickleorder:", trickleorder
		for currtrickler in trickleorder:
			#print "||trickling:", currtrickler
			self.TrickleQuantity(currtrickler, ChangedVariables)
			#print "||configuration after that trickling:", self.quantitytrickledvalues
		self.quantitiesshifted = True
		#print "|RESULT:", self.quantitytrickledvalues
	def TrickleQuantity(self, name, ChangedVariables):
		#print "|||trickling", name
		for effect in QuantityDefinitions[name].QuantityEffects:
			self.ApplyQuantityTricklingEffect(effect, name)
			#print "|||after applying trickling effect", effect, "quantitytrickledvalues is:", self.quantitytrickledvalues
		for annuity in QuantityDefinitions[name].QuantityAnnuityEffects:
			self.quantitytrickledvalues["direct_value"] += annuity.Value([], VariableDefinitions, self, ChangedVariables)
			# note: remember what annuity.Value actually DOES! It does not give the value of a single instance of the annuity. It does not give some sort of "time-flattened to present" value that we can just add to quantitytrickledvalues[annuity.quantname]. Rather, it returns the ULTIMATE VALUE of the annuity, ie how much the annuity contributes to direct_value. For determining that, it looks at the value of a single annuity-payout now and assumes every future annuity-payout will have the same payout. We DO NOT have to multiply annuity.Value by quantitytrickledvalues[name].
			#print "|||after applying /annuity/ trickling effect on line", annuity.linenum, " quantitytrickledvalues is:", self.quantitytrickledvalues
	def ApplyQuantityTricklingEffect(self, Effect, Originator, ChangedVariables={}):
		# NOTE: items in self.val are DISTRIBUTIONS; items in quantitytrickledvalues are FLOATS
		#print "applying quantity trickling from ", Originator, "to", Effect.quantname, "; at the start it's ", self.quantitytrickledvalues[Effect.quantname]
		quantnumtransform = Effect.effect
		#print "calling quantnumtransform.mean, bottomcurr=" + str(self.quantitytrickledvalues[Effect.quantname]) + " , toppost=" + str(self.quantitytrickledvalues[Originator])
		#print "originator:",self.quantitytrickledvalues[Originator]
		newamt = quantnumtransform.mean(VariableDefinitions, self, ChangedVariables, bottomcurr=DegenerateDist(self.quantitytrickledvalues[Effect.quantname]), toppost=DegenerateDist(self.quantitytrickledvalues[Originator]))
		#print "newamt:", newamt
		#for bottomcurr in (0,1,2,3):
		#	for toppost in (0,1,2,3):
		#		print "quantnumtransform under",bottomcurr,toppost,"is: ",quantnumtransform.mean(bottomcurr=bottomcurr,toppost=toppost)
		#print ">>>>>>>>>>"
		#print quantnumtransform
		#print self.quantitytrickledvalues
		#print "newamt: ", newamt

		#newamt = self.quantitytrickledvalues[Effect.quantname] + 6*(self.quantitytrickledvalues[Originator])
		#TODO: make this actually take into account the actual function, rather than just making it 6*
		#self.approxcontributions[Effect.quantname][Originator] = self.approxcontributions[Effect.quantname].get(Originator, 0.) + newamt - self.quantitytrickledvalues[Effect.quantname] #new minus old equals change
		#print "...and at the end it's ", newamt
		#print "newamt type:", type(newamt).__name__
		self.quantitytrickledvalues[Effect.quantname] = newamt
		# for reference for todo:
		#quantnumtransform = CreateFunction(QuantityDefinitions[name], "quantity-quanteffect") #this is the function for transforming the trickledvalue of the top thing (from which we're trickling) to an amount to add to the trickledvalue of the bottom thing
		#transformedquantity = quantnumtransform(self.quantitytrickledvalues[currquantity], self.quantitydefs[subline.quantity].curramount, self.quantitytrickledvalues[subline.quantity])
	def ApplyQuantityStraightEffect(self, Effect, ChangedVariables):
		assert Effect.quantname in self.val, "ERROR: trying to apply straight effect to quantity, but quantity is unknown; effect is " + str(Effect)
		quantnumtransform = Effect.effect
		newamt = quantnumtransform.mean(VariableDefinitions, self, ChangedVariables, bottomcurr=self.val[Effect.quantname])
		#TODO: make this actually construct the resulting distribution! at the moment it forces the quantity's distribution to become degenerate
		self.val[Effect.quantname] = DegenerateDist(newamt)
		self.Rustle()

def ShowSummary(Config):
	print; print "############################CONFIGURATION SUMMARY##########################"
	print; print "Quantity Value:", Config.GetQuantityValue()
	AgosData = []
	for agosname in Config.ExecutableAgosi:
		V = Config.AgosIndirectValue(agosname)
		AgosData.append((V[0], agosname, V))
	AgosData.sort(reverse=True)
	for code, agosname, V in AgosData:
		print; print "    *******" + agosname + "*******"
		print str(V[3])
		#ACCURACY = 0
		#print str(round(V[1], ACCURACY)) + "u\t/\t" + str(round(V[2]/3600.,ACCURACY)) + "h\t=\t" + str(round(V[0]*3600., ACCURACY)) + "u/h"
		print str(int(V[1])) + "u\t/\t" + str(int(V[2]/3600.)) + "h\t=\t" + str(int(V[0]*3600.)) + "u/h"
	print
	for varname in VariableNames:
		print varname, Config.VariableEVOPI(varname) #TODO also print value-rate of information-gathering
	print





if __name__ == "__main__":
	MainConfig = ParseConfig("configuration.txt", isfile=True)
	ShowSummary(MainConfig)



