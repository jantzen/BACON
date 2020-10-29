from BACON import *


def test_Constant_test_condition():
    t1 = Term('t1', np.array([1, 2, 3],dtype='float'))
    t2 = Term('t2', np.array([1, 1, 0.9],dtype='float'))
    t3 = Term('t3', np.array([1, 1, 0.8],dtype='float'))

    table = Table([t1, t2, t3])

    delta = 0.1

    const_rule = Constant(delta)

    out = const_rule.test_condition(table)

    assert out[0] == t2


def test_Constant_apply_rule():
    t1 = Term('t1', np.array([1, 2, 3],dtype='float'))
    t2 = Term('t2', np.array([1, 1, 0.9],dtype='float'))
    t3 = Term('t3', np.array([1, 1, 0.8],dtype='float'))

    table = Table([t1, t2, t3])

    delta = 0.1

    const_rule = Constant(delta)

    out = const_rule.test_condition(table)

    const_rule.apply_rule(table, out)

    assert ('t2 = c' in table._laws)


def test_Linear_is_linear():
    t1 = Term('t1', np.arange(0.,10., 0.1))
    t2 = Term('t2', np.random.normal(0., 1., (100,)))
    t3 = Term('t3', t1._values + np.random.normal(0., 0.05, (100,)))


    delta = 0.1

    linear_rule = Linear(delta)

    neg = linear_rule.is_linear(t1._values, t2._values)

    pos = linear_rule.is_linear(t1._values, t3._values)

    assert not neg 
    assert pos


def test_Linear_test_condition():
    t1 = Term('t1', np.arange(0.,10., 0.1))
    t2 = Term('t2', np.random.normal(0., 1., (100,)))
    t3 = Term('t3', t1._values + np.random.normal(0., 0.05, (100,)))

    table = Table([t1, t2, t3])

    delta = 0.1

    linear_rule = Linear(delta)

    out = linear_rule.test_condition(table)

    assert out[0] == t1 and out[1] == t3


def test_Linear_apply_rule():

    t1 = Term('t1', np.array([1, 2, 3],dtype='float'))
    t2 = Term('t2', np.array([1, 2, 2.7],dtype='float'))
    t3 = Term('t3', np.array([3, 5, 6.8],dtype='float'))

    table = Table([t1, t2, t3])

    delta = 0.1

    linear_rule = Linear(delta)

    out = linear_rule.test_condition(table)

    linear_rule.apply_rule(table, out)

    assert len(table._laws) == 1


def test_Increasing_is_increasing():
    t1 = Term('t1', np.arange(0.,10., 0.1))
    t2 = Term('t2', np.random.normal(0., 1., (100,)))
    t3 = Term('t3', t1._values**2 + np.random.normal(0., 0.05, (100,)))


    delta = 0.1

    increasing_rule = Increasing(delta)

    neg = increasing_rule.is_increasing(t1._values, t2._values)

    pos = increasing_rule.is_increasing(t1._values, t3._values)

    assert not neg 
    assert pos


def test_Increasing_test_condition():
    t1 = Term('t1', np.arange(0., 10., 0.1))
    t2 = Term('t2', np.random.normal(0., 1., (100,)))
    t3 = Term('t3', t1._values**2 + np.random.normal(0., 0.05, (100,)))

    table = Table([t1, t2, t3])

    delta = 0.1

    increasing_rule = Increasing(delta)

    out = increasing_rule.test_condition(table)

    assert out[0] == t1 and out[1] == t3


def test_Increasing_apply_rule():
    t1 = Term('t1', np.arange(0.,10., 0.1))
    t2 = Term('t2', np.random.normal(0., 1., (100,)))
    t3 = Term('t3', t1._values**2 + np.random.normal(0., 0.05, (100,)))

    table = Table([t1, t2, t3])

    delta = 0.1

    increasing_rule = Increasing(delta)

    out = increasing_rule.test_condition(table)
    increasing_rule.apply_rule(table, out)
    assert len(table._terms) == 4


def test_Decreasing_is_decreasing():
    t1 = Term('t1', np.arange(0.1, 10., 0.1))
    t2 = Term('t2', np.random.normal(0., 1., (99,)))
    t3 = Term('t3', 1. / t1._values + np.random.normal(0., 0.05, (99,)))

    delta = 0.1

    decreasing_rule = Decreasing(delta)

    neg = decreasing_rule.is_decreasing(t1._values, t2._values)

    pos = decreasing_rule.is_decreasing(t1._values, t3._values)

    assert not neg 
    assert pos


def test_Decreasing_test_condition():
    t1 = Term('t1', np.arange(0.1, 10., 0.1))
    t2 = Term('t2', np.random.normal(0., 1., (99,)))
    t3 = Term('t3', 1. / t1._values + np.random.normal(0., 0.05, (99,)))

    delta = 0.1

    decreasing_rule = Decreasing(delta)

    table = Table([t1, t2, t3])

    out = decreasing_rule.test_condition(table)

    assert out[0] == t1 and out[1] == t3


def test_Decreasing_apply_rule():
    t1 = Term('t1', np.arange(0.1, 10., 0.1))
    t2 = Term('t2', np.random.normal(0., 1., (99,)))
    t3 = Term('t3', 1. / t1._values + np.random.normal(0., 0.05, (99,)))

    delta = 0.1

    decreasing_rule = Decreasing(delta)

    table = Table([t1, t2, t3])

    out = decreasing_rule.test_condition(table)

    decreasing_rule.apply_rule(table, out)

    assert len(table._terms) == 4 and table._terms[3]._definition == 't1 * t3'


def test_find_laws():
    t1 = Term('t1', np.arange(0.1, 10., 0.1))
    t2 = Term('t2', np.random.normal(0., 1., (99,)))
    t3 = Term('t3', 1. / t1._values + np.random.normal(0., 0.05, (99,)))

    delta = 0.1

    decreasing_rule = Decreasing(delta)

    table = Table([t1, t2, t3])

    rules = [Constant(delta), Linear(delta), Increasing(delta),
            Decreasing(delta)]

    out = table.find_laws(rules)

    assert len(out) > 0


def test_load_data():
    pass

def test_build_chart():
#    t1 = Term('battery', ['A','B','C','A','B','C','A','B','C'])
#    t2 = Term('wire', ['X','X','X','Y','Y','Y','Z','Z','Z'])
#    t3 = Term('current', [3.4763, 3.9781, 5.5629, 4.8763, 5.5803, 7.8034,
#        3.0590, 3.5007, 4.8952])
    t1 = Term('battery', ['A','A','A','B','B','B','C','C','C'])
    t2 = Term('wire', ['X','Y','Z','X','Y','Z','X','Y','Z'])
    t3 = Term('current', np.array([3.4763, 4.8763, 3.0590, 3.9871, 5.5803, 3.5007,
        5.5629, 7.8034, 4.8952]))

    chart = build_chart([t1,t2],t3)

    header = ['battery', 'wire', 'current']

    data = [['A','X',3.4763],['A','Y',4.8763],['A','Z',3.0590],['B','X',3.9871],
            ['B','Y',5.5803],['B','Z',3.5007],['C','X',5.5629],['C','Y',7.8034],['C','Z',4.8952]]
    
    assert [header, data] == chart


def test_find_blocks():
    data = [['A','A','A',1],['A','A','B',2]]
    header = ['column1','column2']
    chart = [header, data]
    blocks = find_blocks(chart)


    assert blocks == [[0,1]]

    t1 = Term('battery', ['A','A','A','B','B','B','C','C','C'])
    t2 = Term('wire', ['X','Y','Z','X','Y','Z','X','Y','Z'])
    t3 = Term('current', np.array([3.4763, 4.8763, 3.0590, 3.9871, 5.5803, 3.5007,
        5.5629, 7.8034, 4.8952]))

    chart = build_chart([t1,t2],t3)

    blocks = find_blocks(chart[1])

    assert blocks == [[0,2],[3,5],[6,8]]

def test_constancy():
    values = np.linspace(0., 10., 5.)
    assert not constancy(values)

    values = np.random.normal(0., size=(10))

def test_invent_variables():
    t1 = Term('battery', ['A','A','A','B','B','B','C','C','C'])
    t2 = Term('wire', ['X','Y','Z','X','Y','Z','X','Y','Z'])
    t3 = Term('current', np.array([3.4763, 4.8763, 3.0590, 3.9871, 5.5803, 3.5007,
        5.5629, 7.8034, 4.8952]))
    table = Table()
    table.add_term(t1)
    table.add_term(t2)
    table.add_term(t3)

    table.invent_variables()

    added_term = table._terms[-1]
    
    expected = [3.4763, 4.8763, 3.0590,3.4763, 4.8763, 3.0590, 3.4763, 4.8763, 3.0590]
    
    assert expected == added_term._values
