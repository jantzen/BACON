from BACON import *

table_boyle = Table()

table_boyle.load_data('../phil6334/data/ohm_law_data_neon.pkl')

c, i, d, l = (Constant(), Increasing(), Decreasing(), Linear())

table_boyle.invent_variables()
