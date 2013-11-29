import unittest
from worlds import *

SimpleConfig = """
Q direct_value 10

A make_value
	now 1h
	direct_value 100
"""

TreeConfig = """
Q direct_value 1

Q A 1
	direct_value 1

Q B 1
	direct_value 1

Q AA 1
	A 1

Q AB 1
	A 1
	B 1

Q BB 1
	B 1

"""

MetaAnnuityConfig = """
Q direct_value 1

Q dv_annuity 1
	@direct_value 2 r1

Q dv_meta_annuity 1
	@dv_annuity 2 r1

Q dv_meta_meta_annuity 1
	@dv_meta_annuity 2 r1

A make_dv_mm_annuity
	now 1h
	dv_meta_meta_annuity 1
"""

class TestQuantityValuation(unittest.TestCase):
	def test_simple(self):
		self.assertEqual(ParseConfig(SimpleConfig).GetQuantityValue(), 10.)
	def test_meta_annuities(self):
		C = ParseConfig(MetaAnnuityConfig)
		shouldbe = 2*2*2+2*2+2+1
		self.assertTrue((shouldbe-.1) < C.GetQuantityValue() < (shouldbe+0.01))
	def test_tree_propagation(self):
		C = ParseConfig(TreeConfig)
		shouldbe = 1 + 1*(1+(1*1)+(1*1)) + 1*(1+(1*1)+(1*1))
		self.assertTrue((shouldbe-.1) < C.GetQuantityValue() < (shouldbe+0.01))

class TestAgosValuation(unittest.TestCase):
	def test_simple(self):
		C = ParseConfig(SimpleConfig)
		self.assertTrue(abs(
			C.AgosIndirectValue("make_value")[0] - 100./3600.)
			< 0.01 )
	def test_meta_annuity_creation(self):
		C = ParseConfig(MetaAnnuityConfig)
		self.assertTrue(abs(
			C.AgosIndirectValue("make_dv_mm_annuity")[0] - (2*2*2)/3600.)
			< 0.1 )

if __name__ == "__main__":
	unittest.main()
