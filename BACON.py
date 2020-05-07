# BACON.py

import numpy as np
import pdb
from scipy import stats
import pickle
import scipy.stats
import pandas as pd

def build_chart(nominal_terms, quantitative_term):
    """ This is a helper function for the invent_variables method.
    Returns a pd DataFrame where columns are Term values, col names are definitions
    nominal_terms: list of Terms
    quantitative_term: one Term
    """
    chart = pd.DataFrame()
    nom_names = []

    # set index so we can add columns
    chart.Index = len(nominal_terms[0]._values)

    # add nominal columns
    for i in range(len(nominal_terms)):
        chart[nominal_terms[i]._definition] = nominal_terms[i]._values
        nom_names.append(nominal_terms[i]._definition)

    # add quant column
    chart[quantitative_term._definition] = quantitative_term._values

    # sort by nominal terms
    chart = chart.sort_values(nom_names)
    
    # reset index
    chart.index = pd.RangeIndex(len(chart.index))
    
    return chart


def find_blocks(data, depth=1):
    """ This is a helper function for the invent_variables method.
    It finds "blocks," rows where all the qualitative vars (except the ones we
    ignore) are the same, and returns start and end index for each block.
    data:   the chart dataframe
    depth:  the number of qualitative variables to ignore when finding blocks
    """
    blocks = []
    start = 0
    end = 0
    d = depth + 1

    # subset of data w/o last d columns
    data = data.iloc[:,:-d]
    print("find_blocks data", data)
    previous_row = data.iloc[0,:]

    pdb.set_trace()
    for i, row in data.iterrows():
        current_row = row
        if not current_row.equals(previous_row):
            # save ranges for prior block, and reset
            end = i-1
            blocks.append([start,end])
            start = i
        previous_row = current_row.copy()
    end = len(data) - 1
    blocks.append([start, end])

    return blocks


def constancy(values, threshold=0.5):
#    values = np.array(values)
#    [W, p] = stats.shapiro(values)
#    if p < threshold:
#        return False
#    else:
#        return True
    values = np.array(values)
    print("values", values)
    z_scores = np.array(values - np.mean(values))/np.std(values)
    print("z_scores", z_scores)
    z_max = np.max(z_scores)
    p = scipy.stats.norm.sf(abs(z_max))
    if p < threshold:
        return False
    else:
        return True


class Term( object ):

    """ Attributes:
        definition: string
        values: numpy array with shape = (#samples, )

        Methods:
        none
    """

    def __init__(self, definition, values):
        
        self._definition = definition
        self._values = values

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
            and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)


class Table( pd.DataFrame ):
    
    """ Attributes:
        terms: a list of Term objects
        laws: a set of laws (strings)

        Methods:
        add_law
        count_laws
        add_term
        find_laws
        load_data
        invent_variables
    """

    _metadata = ['added_property'] # don't think we need this

    @property
    def _constructor(self):
        return Table

    def __init__(self, terms=None, laws=None):
        super().__init__()
        if terms == None:
            self._terms = []
        else:
            self._terms = terms

        if laws == None:
            self._laws = set([])
        else:
            self._laws = laws

    def add_law(self, law):
        if law not in self._laws:
            self._laws.add(law)

    def count_laws(self):
        return len(self._laws)

    def add_term(self, term):
        self._terms.append(term)

    def find_laws(self, rules, n=1, max_iterations=100):
        """ Given a list of Rule objects, returns the first n laws
        discovered, or None if max_iterations is reached>
        """
    
        for i in range(max_iterations):
            for rule in rules:
                terms = None
                terms = rule.test_condition(self)
    
                if terms != None:
                    rule.apply_rule(self, terms)
    
                if self.count_laws() >= n:
                    return self._laws

    def load_data(self, filename, var_defs=None):
        """ Loads pickled data as terms. Data is assumed to be a list in which 
        each element is a list representing a single sample (a value for each
        variable). var_defs is a list of strings naming the imported terms.
        """
        f = open(filename, 'rb')
        data = pickle.load(f)
        f.close()

        if var_defs == None:
            var_defs = []
            num_vars = len(data[0])
            for i in range(num_vars):
                var_defs.append('t' + str(i))
        
        # transpose the data
        cols = []
        for col_index in range(num_vars):
            column = []
            for row in data:
                column.append(row[col_index])
            cols.append(column)
        data = cols

        # add terms
        for index, row in enumerate(data):
            term = Term(var_defs[index], row)
            self.add_term(term)



    def invent_variables(self):

        """ This method is temporarily restricted to considering two nominal
        variables and a single quantitative variable. It is also artificially
        restricted to assuming linear relations amongst the invented variables.
        """
        
        # find all of the nominal terms (if any), and stack them up
        nominal_terms = []
        quantitative_terms = []

        for term in self._terms:
            values = term._values
            if type(values[0]) == str:
                nominal_terms.append(term)
            else:
                quantitative_terms.append(term)

        if len(quantitative_terms) > 1:
            raise ValueError("This version of the 'invent_variables' method cannot handle multiple quantitative variables.")

#        if len(nominal_terms) == 2:

#            chart = build_chart(nominal_terms, quantitative_terms[0])
#
#            start = 0
#            
#            # find index range of first block
#            end = np.where(chart[:,0] == chart[0,0])[0][-1]
#
#            # decide whether there is variation within that block
#            M = np.mean(chart[start:end, 2])
#            std = np.std(chart[start:end, 2])

        # find blocks
        # for each block
            # test for variance
            # if variance, introduce a new intrinsic variable, fill in values
            # for all rows, and exit for loop

        # if any new intrinsic variables were added, then for each block
            # do a linear fit, add both slope and intercept as terms

        chart = build_chart(nominal_terms, quantitative_terms[0])

        print(chart)

        blocks = find_blocks(chart)

        no_trends = True
        assignment = dict([])

        for block in blocks:
            if no_trends == True:
                print(block)
                data = quantitative_terms[0]._values[block[0]:block[1]]
                print(data)
                if not constancy(data):
                    no_trends = False
                    # build key:
                    for i in range(block[0],block[1]+1):
                        key = nominal_terms[-1]._values[i]
                        value = quantitative_terms[0]._values[i]
                        assignment.update([(key,value)])
                    
        if not len(assignment) == 0:
            # build a new term (corresponing to new intrinsic property)
            new_values = []
            for qual in nominal_terms[-1]._values:
                new_values.append(assignment[qual])
            definition = 'intrinsic1'
            new_term = Term(definition, new_values)
            self.add_term(new_term)

        



class ProductionRule( object ):

    """ Attributes:
        delta

        Methods:
        test_condition
        apply_rule
    """
    
    def __init__(self, delta=0.1):
        self._delta = delta


    def test_condition(self, table):
        pass


    def apply_rule(self, table, terms):
        pass


class Constant( ProductionRule ):

    def test_condition(self, table):
        
        #returns the first term that satisfies the condition or 'None' otherwise
        out = None
        
        for i in range(len(table._terms)):
            
            if type(table._terms[i]._values[0]) == str:
                break

            M = np.mean(table._terms[i]._values)
            sigma = np.std(table._terms[i]._values)

            constant = True

            if sigma / M > self._delta:
                constant = False

            if constant == True:
                return [table._terms[i]]

        return None

    def apply_rule(self, table, terms):
        law = str(terms[0]._definition + ' = c')
        table.add_law(law)


class Linear( ProductionRule ):
    
    def is_linear(self, values1, values2):
        slope, intercept, r_value, p_value, std_err = stats.linregress(values1,
                values2)

        if r_value ** 2 > 1 - self._delta:
            return True
       
        else:
            return False
        
        return linear


    def test_condition(self, table):
        """ returns a list of the first two terms that satisfy the condition or 
            'None' otherwise 
        """

        # loop over all combinations of terms until satisfied or combinations are
        # exhausted

        for i in range(len(table._terms) - 1):
            for j in range(i+1, len(table._terms)):
                if (type(table._terms[i]._values[0]) == str or
                        type(table._terms[j]._values[0]) == str):
                    break
           
                if self.is_linear(table._terms[i]._values,
                   table._terms[j]._values):
                      return [table._terms[i], table._terms[j]]

        return None


    def apply_rule(self, table, terms):
        """ fits a line to terms and reports this as a law.
        """

        # fit the line
        x = terms[0]._values
        y = terms[1]._values
        A = np.vstack([x, np.ones(len(x))]).T
        m, b = np.linalg.lstsq(A, y)[0]

        law = str(terms[1]._definition + ' = ' + str(m) + ' * ' +
                terms[0]._definition + ' + ' + str(b))
        table.add_law(law)


class Increasing( ProductionRule ):

    def is_increasing(self, values1, values2):
        x = np.abs(values1)
        y = np.abs(values2)
       
        tau, p_value = stats.kendalltau(x, y)

        if p_value < 0.05 and tau > 0:
            return True

        else:
            return False


    def test_condition(self, table):
        """ returns a list of the first two terms that satisfy the condition or 
            'None' otherwise 
        """

        # loop over all combinations of terms until satisfied or combinations are
        # exhausted

        for i in range(len(table._terms) - 1):
            for j in range(i+1, len(table._terms)):
 
                if (type(table._terms[i]._values[0]) == str or
                    type(table._terms[j]._values[0]) == str):
                    break
            
                if self.is_increasing(table._terms[i]._values,
                    table._terms[j]._values):
                        return [table._terms[i], table._terms[j]]

        return None

        
    def apply_rule(self, table, terms):
        definition = str(terms[0]._definition + ' / ' + terms[1]._definition)
        x = terms[0]._values
        y = terms[1]._values
        values = np.zeros(len(x))
        for i in range(len(x)):
            if not x[i] == 0.:
                values[i] = y[i] / x[i]
            else:
                values[i] = np.inf
 
        new_term = Term(definition, values)

        table._terms.append(new_term)


class Decreasing( ProductionRule ):

    def is_decreasing(self, values1, values2):
        x = np.abs(np.array(values1))
        y = np.abs(np.array(values2))
 
        tau, p_value = stats.kendalltau(x, y)

        if p_value < 0.05 and tau < 0:
            return True

        else:
            return False


    def test_condition(self, table):
        """ returns a list of the first two terms that satisfy the condition or 
            'None' otherwise 
        """

        # loop over all combinations of terms until satisfied or combinations are
        # exhausted

        for i in range(len(table._terms) - 1):
            for j in range(i+1, len(table._terms)):
                if (type(table._terms[i]._values[0]) == str or
                       type(table._terms[j]._values[0]) == str):
                    break
            
                if self.is_decreasing(table._terms[i]._values,
                    table._terms[j]._values):
                        return [table._terms[i], table._terms[j]]

        return None

        
    def apply_rule(self, table, terms):
        definition = str(terms[0]._definition + ' * ' + terms[1]._definition)
        x = np.array(terms[0]._values)
        y = np.array(terms[1]._values)
        values = y * x

        new_term = Term(definition, values)

        table._terms.append(new_term)



