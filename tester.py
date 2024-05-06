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
        self.table_boyle = Table()
        self.table_boyle.load_data('../phil6334/data/ohm_law_data_neon.pkl')
   
    def test_invent_variables(self):
        self.table_boyle.invent_variables()
        self.assertEqual("intrinsic1", self.table_boyle._terms[3]._definition, "There should \
                be a fourth term now")




#c, i, d, l = (Constant(), Increasing(), Decreasing(), Linear())


if __name__ == '__main__':
    unittest.main()
