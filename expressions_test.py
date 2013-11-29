import unittest
from expressions import *

class TestDistributionCreation(unittest.TestCase):
	def test_nested_arguments(self):
		self.assertEqual(MakeDistribution("norm(norm(norm(1,1),1),1)").mean({},{},{}), 1)
	def test_right_association(self):
		self.assertEqual(MakeDistribution("1+2*3+4*5+6").mean({},{},{}), 95)
	def test_right_association(self):
		self.assertEqual(MakeDistribution("1+2*3+4*5+6").mean({},{},{}), 95)

class TestNormDist(unittest.TestCase):
	def test_rep_mean(self):
		reps = MakeDistribution("norm(norm(norm(1,1),1),1)").representatives({},{},{})
		self.assertTrue(0.99 < sum([i*reps[i] for i in reps.keys()]) < 1.01)

class TestUniformDist(unittest.TestCase):
	def test_mean(self):
		self.assertEqual(MakeDistribution("unif(11,17)").mean({},{},{}), (11.+17.)/2.)
	def test_rep_mean(self):
		reps = MakeDistribution("unif(11,17)").representatives({},{},{})
		themean = MakeDistribution("unif(11,17)").mean({},{},{})
		self.assertTrue((themean-.01) < sum([i*reps[i] for i in reps.keys()]) < (themean+.01))
	def test_compound_dist_rep_mean(self):
		reps = MakeDistribution("unif(11,17)*norm(10,5)").representatives({},{},{})
		themean = MakeDistribution("unif(11,17)*norm(10,5)").mean({},{},{})
		self.assertTrue((themean-.01) < sum([i*reps[i] for i in reps.keys()]) < (themean+.01))

class TestMeanFunc(unittest.TestCase):
	def test_compound_mean(self):
		self.assertTrue( # assert that the error is small (between what the mean function gives and what direct calculation gives)
			abs(
				MakeDistribution("mean(unif(11,17),norm(10,5),norm(1,17))").mean({},{},{}) -
				sum( [ MakeDistribution(s).mean({},{},{}) for s in ("unif(11,17)", "norm(10,5)", "norm(1,17)") ] ) / 3.
				)
			< 0.01)

if __name__ == "__main__":
	unittest.main()
