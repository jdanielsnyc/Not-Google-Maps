#!/usr/bin/env python3
import os
import lab
import pickle
import pytest

TEST_DIRECTORY = os.path.dirname(__file__)

def make_dataset_loader():
    cache = {}
    def load_dataset(name):
        if name not in cache:
            nodes_name = os.path.join(TEST_DIRECTORY, 'resources', f'{name}.nodes')
            ways_name = os.path.join(TEST_DIRECTORY, 'resources', f'{name}.ways')
            cache[name] = lab.build_auxiliary_structures(nodes_name, ways_name)
        return cache[name]
    return load_dataset

load_dataset = make_dataset_loader()


def _tuple_close(t1, t2):
    return (len(t1) == len(t2)
            and all(abs(i - j) <= 1e-9 for i, j in zip(t1, t2)))

def compare_output(name, inputs, test_num, type_, nodes=False):
    aux = load_dataset(name)
    exp_fname = f'test_data/test_{name}_{test_num:02d}_{type_}{"_nodes" if nodes else ""}.pickle'
    with open(exp_fname, 'rb') as f:
        expected_path = pickle.load(f)
    compare_result_expected(aux, inputs, expected_path, type_, nodes)


def compare_result_expected(aux, inputs, expected_path, type_, nodes=False):
    if type_ == 'short':
        test_func = lab.find_short_path_nodes if nodes else lab.find_short_path
    else:
        test_func = lab.find_fast_path
    result_path = test_func(aux, *inputs)
    if expected_path is None:
        assert result_path is None
    else:
        assert len(result_path) == len(expected_path), 'Path lengths differ.'
        all_good = True
        for ix, (v1, v2) in enumerate(zip(result_path, expected_path)):
            if isinstance(v2, tuple):
                if not _tuple_close(v1, v2):
                    all_good = False
                    break
            else:
                if v1 != v2:
                    all_good = False
                    break
        assert all_good, f'Paths differ at position {ix}.  Expected {v2}, got {v1}.'


def test_mit_short_nodes_00():
    # Should take the most direct path: New House, Kresge, North Maseeh, Lobby
    # 7, Building 26, 34-501
    loc1 = 2 # New House
    loc2 = 8 # 34-501
    expected_path = [2, 1, 10, 5, 6, 8]
    compare_result_expected(load_dataset('mit'), (loc1, loc2), expected_path, 'short', True)

def test_mit_short_nodes_01():
    # Should take path Building 35, Lobby 7, North Maseeh, South Maseeh
    # Tests that edges only connect consecutive nodes on a way
    loc1 = 7 # near Building 35
    loc2 = 3 # Near South Maseeh
    expected_path = [7, 5, 10, 3]
    compare_result_expected(load_dataset('mit'), (loc1, loc2), expected_path, 'short', True)

def test_mit_short_nodes_02():
    # Should take path Building 35, Lobby 7, North Maseeh, South Maseeh
    # Tests that edges only connect consecutive nodes on a way
    loc1 = 1 # Kresge
    loc2 = 2 # New House
    expected_path = [1, 10, 3, 2]
    compare_result_expected(load_dataset('mit'), (loc1, loc2), expected_path, 'short', True)

def test_mit_short_nodes_03():
    # Both nodes are valid, but there is no path that connects them.
    loc1 = 11
    loc2 = 10
    compare_result_expected(load_dataset('mit'), (loc1, loc2), None, 'short', True)


@pytest.mark.parametrize('testcase', [(0, 272855431, 233945564), (1, 233985606, 234006596), (2, 5260415861, 234038557)])
def test_midwest_short_nodes(testcase):
    ix, start, end = testcase
    compare_output('midwest', (start, end), ix, 'short', True)


CAMBRIDGE_NODE_TESTS = [(61321294, 567774187), (61321294, 61328038),
                        (61328038, 61321294), (5458770478, 2484568633),
                        (1281652591, 61385480), (61385480, 1281652591)]
@pytest.mark.parametrize('testcase', list(enumerate(CAMBRIDGE_NODE_TESTS)))
def test_cambridge_short_nodes(testcase):
    ix, nodes = testcase
    compare_output('cambridge', nodes, ix, 'short', True)



def test_mit_short_00():
    # Should take the most direct path: New House, Kresge, North Maseeh, Lobby 7, Building 26, 34-501
    loc1 = (42.355, -71.1009) # New House
    loc2 = (42.3612, -71.092) # 34-501
    expected_path = [
        (42.355, -71.1009), (42.3575, -71.0952), (42.3582, -71.0931),
        (42.3592, -71.0932), (42.36, -71.0907), (42.3612, -71.092),
    ]
    compare_result_expected(load_dataset('mit'), (loc1, loc2), expected_path, 'short')

def test_mit_short_01():
    # Should take path Building 35, Lobby 7, North Maseeh, South Maseeh
    # Tests that edges only connect consecutive nodes on a way
    loc1 = (42.3603, -71.095) # near Building 35
    loc2 = (42.3573, -71.0928) # Near South Maseeh
    expected_path = [
        (42.3601, -71.0952), (42.3592, -71.0932),
        (42.3582, -71.0931), (42.3575, -71.0927),
    ]
    compare_result_expected(load_dataset('mit'), (loc1, loc2), expected_path, 'short')

def test_mit_short_02():
    # Should take path Kresge, North Maseeh, South Maseeh, New House
    # Tests that one-ways are only allowed to go in certain direction
    loc1 = (42.3576, -71.0952) # Kresge
    loc2 = (42.355, -71.1009) # New House
    expected_path = [
        (42.3575, -71.0952), (42.3582, -71.0931),
        (42.3575, -71.0927), (42.355, -71.1009),
    ]
    compare_result_expected(load_dataset('mit'), (loc1, loc2), expected_path, 'short')

def test_mit_short_03():
    # Should take path Kresge, North Maseeh, Lobby 7, Building 26
    # Tests that for non-exact
    # Tests that nodes that aren't in any way are not used
    loc1 = (42.3576, -71.0951) #close to Kresge
    loc2 = (42.3605, -71.091) # is near an invalid node: Unreachable Node
    expected_path = [
        (42.3575, -71.0952), (42.3582, -71.0931),
        (42.3592, -71.0932), (42.36, -71.0907),
    ]
    compare_result_expected(load_dataset('mit'), (loc1, loc2), expected_path, 'short')

def test_mit_short_04():
    # Should return None
    # Tests node with no outgoing edges
    loc1 = (42.3575, -71.0956) # Parking Lot - end of a oneway and not on any other way
    loc2 = (42.3575, -71.0940) #close to Kresge
    compare_result_expected(load_dataset('mit'), (loc1, loc2), None, 'short')


MIDWEST_TESTS = [
    ((41.375288, -89.459541), (41.452802, -89.443683)),
    ((41.505515, -89.463392), (41.43567, -89.394277)),
    ((41.367973, -89.478311), (41.446346, -89.317066)),
]
@pytest.mark.parametrize('testcase', list(enumerate(MIDWEST_TESTS)))
def test_midwest_short(testcase):
    ix, inps = testcase
    compare_output('midwest', inps, ix, 'short')


CAMBRIDGE_TESTS = [
    ((42.359242, -71.093765), (42.358984, -71.114862)),
    ((42.359242, -71.093765), (42.360485, -71.108349)),
    ((42.360485, -71.108349), (42.359242, -71.093765)),
    ((42.403524, -71.23408), (42.348838, -71.093667)),
    ((42.336, -71.1678), (42.3398, -71.1063)),
    ((42.3398, -71.1063), (42.336, -71.1678)),

]
@pytest.mark.parametrize('testcase', list(enumerate(CAMBRIDGE_TESTS)))
def test_cambridge_short(testcase):
    ix, inps = testcase
    compare_output('cambridge', inps, ix, 'short')



def test_mit_fast_00():
    # Should take the a longer, but faster path: New House, Kresge, North Maseeh, Lobby 7, Building 35, 34-501
    loc1 = (42.355, -71.1009) # New House
    loc2 = (42.3612, -71.092) # 34-501
    expected_path = [
        (42.355, -71.1009), (42.3575, -71.0927), (42.3582, -71.0931),
        (42.3592, -71.0932), (42.3601, -71.0952), (42.3612, -71.092),
    ]
    compare_result_expected(load_dataset('mit'), (loc1, loc2), expected_path, 'fast')

def test_mit_fast_01():
    # Should take path Building 26, 34-501, Building 35, Lobby 7
    # Tests that the 'maxspeed_mph' is used instead of highway type speed limit
    # also tests that in the prescence of a repeated way, the highest speed limit is preferred
    loc1 = (42.36, -71.0907) # near Lobby 26
    loc2 = (42.3592, -71.0932) # Near Lobby 7
    expected_path = [
        (42.36, -71.0907), (42.3612, -71.092),
        (42.3601, -71.0952), (42.3592, -71.0932),
    ]
    compare_result_expected(load_dataset('mit'), (loc1, loc2), expected_path, 'fast')

def test_mit_fast_02():
    # Should take path Kresge, North Maseeh, South Maseeh, New House
    # Tests that one-ways are only allowed to go in certain direction
    loc1 = (42.3576, -71.0952) # Kresge
    loc2 = (42.355, -71.1009) # New House
    expected_path = [
        (42.3575, -71.0952), (42.3582, -71.0931),
        (42.3575, -71.0927), (42.355, -71.1009),
    ]
    compare_result_expected(load_dataset('mit'), (loc1, loc2), expected_path, 'fast')

def test_mit_fast_03():
    # Should take path Kresge, North Maseeh, Lobby 7, Building 35, 009 Oh
    # Tests that for non-exact
    # Tests that nodes that aren't in any way are not used
    loc1 = (42.3576 , -71.0951) #close to Kresge
    loc2 = (42.3609, -71.0911) # is near an invalid node: Unreachable Node
    expected_path = [
        (42.3575, -71.0952), (42.3582, -71.0931), (42.3592, -71.0932),
        (42.3601, -71.0952), (42.3612, -71.092),
    ]
    compare_result_expected(load_dataset('mit'), (loc1, loc2), expected_path, 'fast')


@pytest.mark.parametrize('testcase', list(enumerate(MIDWEST_TESTS)))
def test_midwest_fast(testcase):
    ix, inps = testcase
    compare_output('midwest', inps, ix, 'fast')


@pytest.mark.parametrize('testcase', list(enumerate(CAMBRIDGE_TESTS)))
def test_cambridge_fast(testcase):
    ix, inps = testcase
    compare_output('cambridge', inps, ix, 'fast')


if __name__ == '__main__':
    import sys
    import json

    class TestData:
        def __init__(self):
            self.results = {'passed': []}

        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtestloop(self, session):
            yield

        def pytest_runtest_logreport(self, report):
            if report.when != 'call':
                return
            self.results.setdefault(report.outcome, []).append(report.head_line)

        def pytest_collection_finish(self, session):
            self.results['total'] = [i.name for i in session.items]

        def pytest_unconfigure(self, config):
            print(json.dumps(self.results))

    if os.environ.get('CATSOOP'):
        args = ['--color=yes', '-v', __file__]
        if len(sys.argv) > 1:
            args = ['-k', sys.argv[1], *args]
        kwargs = {'plugins': [TestData()]}
    else:
        args = ['-v', __file__] if len(sys.argv) == 1 else ['-v', *('%s::%s' % (__file__, i) for i in sys.argv[1:])]
        kwargs = {}
    res = pytest.main(args, **kwargs)
