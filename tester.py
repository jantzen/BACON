from BACON import *
import unittest

class OhmsLawTestCase(unittest.TestCase):
    """
    Testing whether Bacon can handle data where first qualitative field isn't sorted
    """
    def setUp(self):
        """
        Load Ohm's Law data
        t0  t1  t2
        X   A   0.140
        Y   A   0.168
        Z   A   0.200
        X   B   0.176
        Y   B   0.236
        Z   B   0.288
        X   C   0.292
        Y   C   0.376
        Z   C   0.460
        """
        self.table_ohm = Table()
        self.table_ohm.load_data('ohm_law_data_neon.pkl')
        self.table_ohm.invent_variables()
   
    def test_invent_variables_1(self):
        self.assertEqual("intrinsic_t0", self.table_ohm._terms[3]._definition, "There should \
                be a fourth term now")

    def test_invent_variables_2(self):
        self.assertEqual([0.2, 0.2, 0.2, 0.288, 0.288, 0.288, 0.46, 0.46, 0.46],
                self.table_ohm._terms[3]._values, "First intrinsic should be the \"Z\"s")
        self.assertEqual(5, len(self.table_ohm._terms), "There should be five terms") 
        for term in self.table_ohm._terms:
            print(term)
        # TODO next assertion is not right. Should be slope of t2 vs t3
        self.assertEqual([0.292,0.376,0.46,0.292,0.376,0.46,0.292,0.376,0.46],
                self.table_ohm._terms[4]._values, "Should be the \"C\"s")

    def test_find_laws(self):
        c, i, d, l = (Constant(), Increasing(), Decreasing(), Linear())
        self.table_ohm.find_laws([c,i,d,l])
        print("Laws:", self.table_ohm._laws)
        for term in self.table_ohm._terms:
            print(term)





if __name__ == '__main__':
    unittest.main()
