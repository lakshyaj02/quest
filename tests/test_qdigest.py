from qdigest import QuantileDigest

def test_singleAdd():
    q = QuantileDigest(1)
    q.add(0)
    q.validate()

    assert q.get_confidence_factor() == 0.0
    assert q.get_count() == 1
    assert q.get_node_count() == 1

def test_negative_values():
    q = QuantileDigest(1)
    add_values(q, [-1, -2, -3, -4, -5, 0, 1, 2, 3, 4, 5])

    assert q.get_count() == 11

def test_repeated_value():
    q = QuantileDigest(1)
    add_values(q, [0, 0])

    assert q.get_confidence_factor() == 0.0
    assert q.get_count() == 2
    assert q.get_node_count() == 1

def test_two_distinct_values():
    q = QuantileDigest(1)
    add_values(q, [0, 3])

    assert q.get_confidence_factor() == 0.0
    assert q.get_count() == 2
    assert q.get_node_count() == 3

def test_tree_building():
    q = QuantileDigest(1)
    values = [0, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7]
    add_values(q, values)

    assert q.get_count() == len(values)

def test_tree_building_reverse():
    q = QuantileDigest(1)
    values = [0, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7]
    add_values(q, values)

    assert q.get_count() == len(values)

def test_basic_compression():
    q = QuantileDigest(0.8)
    values = [0, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7]
    add_values(q, values)

    q.compress()
    q.validate()

    assert q.get_count() == len(values)
    assert q.get_node_count() == 7
    assert q.get_confidence_factor() == 0.2

def test_compression():
    q = QuantileDigest(1)

    for i in range(2):
        add_range(q, 0, 15)

    q.compress()
    q.validate()

def test_quantile():
    q = QuantileDigest(1)
    values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    add_values(q, values)

    assert q.get_confidence_factor() == 0.0

    assert q.get_quantiles([0.0]) == [0]
    assert q.get_quantiles([0.1]) == [1]
    assert q.get_quantiles([0.2]) == [2]
    assert q.get_quantiles([0.3]) == [3]
    assert q.get_quantiles([0.4]) == [4]
    assert q.get_quantiles([0.5]) == [5]
    assert q.get_quantiles([0.6]) == [6]
    assert q.get_quantiles([0.7]) == [7]
    assert q.get_quantiles([0.8]) == [8]
    assert q.get_quantiles([0.9]) == [9]
    assert q.get_quantiles([1]) == [9]

def test_quantile_lower_bound():
    q = QuantileDigest(0.5)
    add_range(q, 1, 100)

    assert q.get_quantiles_lower_bound([0.0])[0] == 1
    for i in range(1,11):
        assert q.get_quantiles_lower_bound([i/10.0])[0] <= 10 * i
        if i>5:
            assert q.get_quantiles_lower_bound([i/10.0])[0] >= 10 * (i-5)
    assert q.get_quantiles_lower_bound([0.0, 0.1, 0.2]) == [q.get_quantiles_lower_bound([0.0])[0], q.get_quantiles_lower_bound([0.1])[0], q.get_quantiles_lower_bound([0.2])[0]]

def test_quantile_upper_bound():
    q = QuantileDigest(0.5)
    add_range(q, 1, 100) 

    assert q.get_quantiles_upper_bound([1.0])[0] == 99
    for i in range(1,10):
        assert q.get_quantiles_upper_bound([i/10.0])[0] >= 10 * i
        if i<5:
            assert q.get_quantiles_upper_bound([i/10.0])[0] <= 10 * (i+5)
    assert q.get_quantiles_upper_bound([0.8, 0.9, 1.0]) == [q.get_quantiles_upper_bound([0.8])[0], q.get_quantiles_upper_bound([0.9])[0], q.get_quantiles_upper_bound([1.0])[0]]

def test_weighted_values():
    q = QuantileDigest(1)
    q.add(0,3)
    q.add(2,1)
    q.add(4,5)
    q.add(5,1)
    q.validate()

    assert q.get_confidence_factor() == 0.0

    assert q.get_quantiles([0.0]) == [0]
    assert q.get_quantiles([0.1]) == [0]
    assert q.get_quantiles([0.2]) == [0]
    assert q.get_quantiles([0.3]) == [2]
    assert q.get_quantiles([0.4]) == [4]
    assert q.get_quantiles([0.5]) == [4]
    assert q.get_quantiles([0.6]) == [4]
    assert q.get_quantiles([0.7]) == [4]
    assert q.get_quantiles([0.8]) == [4]
    assert q.get_quantiles([0.9]) == [5]
    assert q.get_quantiles([1]) == [5]

def test_batch_quantile_query():
    q = QuantileDigest(1)
    values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    add_values(q, values)

    assert q.get_confidence_factor() == 0.0

    assert q.get_quantiles([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9]

def test_min_max():
    q = QuantileDigest(0.01)
    add_range(q, 500, 701)

    assert q.get_min() == 500
    assert q.get_max() == 700

def test_equivalence_empty():
    q1 = QuantileDigest(0.01)
    q2 = QuantileDigest(0.01)

    assert q1.equivalent(q2)

def test_equivalence_single_value():
    q1 = QuantileDigest(0.01)
    q2 = QuantileDigest(0.01)

    q1.add(1)
    q2.add(1)

    assert q1.equivalent(q2)

def test_equivalence_single_different_value():
    q1 = QuantileDigest(0.01)
    q2 = QuantileDigest(0.01)

    q1.add(1)
    q2.add(2)

    assert not q1.equivalent(q2)

def test_equivalence_complex():
    q1 = QuantileDigest(0.01)
    q2 = QuantileDigest(0.01)

    add_values(q1, [0, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7])
    add_values(q2, [0, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7])

    assert q1.equivalent(q2)

def test_equivalence_different():
    q1 = QuantileDigest(0.01)
    q2 = QuantileDigest(0.01)

    add_values(q1, [0, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7])
    add_values(q2, [0, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8])

    assert not q1.equivalent(q2)

def test_merge_empty():
    q1 = QuantileDigest(0.01)
    q2 = QuantileDigest(0.01)
    q_pristine = QuantileDigest(0.01)

    q1.merge(q2)

    q1.validate()
    q2.validate()

    assert q1.equivalent(q_pristine)

    assert q1.get_count() == 0
    assert q2.get_count() == 0

    assert q1.get_node_count() == 0
    assert q2.get_node_count() == 0

def test_merge_into_empty():
    q1 = QuantileDigest(0.01)
    q2 = QuantileDigest(0.01)
    q_pristine = QuantileDigest(0.01)

    q2.add(1)
    q_pristine.add(1)

    q1.merge(q2)

    q1.validate()
    q2.validate()

    assert q2.equivalent(q_pristine)

    assert q1.get_count() == 1
    assert q2.get_count() == 1

    assert q1.get_node_count() == 1
    assert q2.get_node_count() == 1

def test_merge_with_empty():
    q1 = QuantileDigest(0.01)
    q2 = QuantileDigest(0.01)
    q_pristine = QuantileDigest(0.01)

    q1.add(1)
    q1.merge(q2)

    q1.validate()
    q2.validate()

    assert q2.equivalent(q_pristine)

    assert q1.get_count() == 1
    assert q2.get_count() == 0

    assert q1.get_node_count() == 1
    assert q2.get_node_count() == 0

def test_merge_sample():
    q1 = QuantileDigest(0.01)
    q2 = QuantileDigest(0.01)

    q1.add(1)
    add_values(q2, [2,3])

    q1.merge(q2)

    q1.validate()

    assert q1.get_count() == 3
    assert q1.get_node_count() == 5

def test_merge_separate_branches():
    q1 = QuantileDigest(0.01)
    q2 = QuantileDigest(0.01)
    pristine_q2 = QuantileDigest(0.01)

    q1.add(1)
    q2.add(2)
    pristine_q2.add(2)

    q1.merge(q2)

    assert q2.equivalent(pristine_q2)

    assert q1.get_count() == 2
    assert q1.get_node_count() == 3

    assert q2.get_count() == 1
    assert q2.get_node_count() == 1

def test_merge_with_lower_level():
    q1 = QuantileDigest(1)
    q2 = QuantileDigest(1)
    pristine_q2 = QuantileDigest(1)

    q1.add(6)
    q1.compress()

    add_values(q2, [0, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 5])
    q2.compress()

    add_values(pristine_q2, [0, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 5])
    pristine_q2.compress()

    q1.merge(q2)

    assert q2.equivalent(pristine_q2)

    assert q1.get_count() == 14
    assert q2.get_count() == 13

def test_merge_with_higher_level():
    q1 = QuantileDigest(1)
    q2 = QuantileDigest(1)
    pristine_q2 = QuantileDigest(1)

    add_values(q1, [0, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 5])
    q1.compress()

    add_values(q2, [6, 7])
    add_values(pristine_q2, [6, 7])

    q1.merge(q2)

    assert q2.equivalent(pristine_q2)

    assert q1.get_count() == 15
    assert q2.get_count() == 2

    assert q1.get_node_count() == 7
    assert q2.get_node_count() == 3

def test_merge_max_level():
    q1 = QuantileDigest(0.01)
    q2 = QuantileDigest(0.01)
    pristine_q2 = QuantileDigest(0.01)

    add_values(q1, [-1, 1])
    add_values(q2, [-2, 2])
    add_values(pristine_q2, [-2, 2])

    q1.merge(q2)

    q1.validate()
    q2.validate()

    assert q2.equivalent(pristine_q2)

    assert q1.get_count() == 4
    assert q1.get_node_count() == 7

def test_merge_same_level():
    q1 = QuantileDigest(1)
    q2 = QuantileDigest(1)
    pristine_q2 = QuantileDigest(1)

    q1.add(0)
    q2.add(0)
    pristine_q2.add(0)

    q1.merge(q2)

    assert q2.equivalent(pristine_q2)

    assert q1.get_count() == 2
    assert q2.get_count() == 1

    assert q1.get_node_count() == 1
    assert q2.get_node_count() == 1

def add_values(q, values):
    for v in values:
        q.add(v)
    q.validate()

def add_range(q, start, end):
    for i in range(start, end):
        q.add(i)
    q.validate()