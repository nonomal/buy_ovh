"""Unit tests for m.catalog.sort_plans.

Backs the interactive sort mode (`s`) and the sortKey/sortReverse conf
keys. Covers the two orderings that matter: numeric columns on their
underlying float, text columns naturally (so 8g precedes 32g)."""

import pytest

from m.catalog import COLUMN_KEYS, sort_plans


def _plan(**kw):
    base = {
        'planCode': '24ska01',
        'model': 'KS-A',
        'cpu': 'N2800',
        'datacenter': 'gra',
        'memory': 'ram-4g-ddr4',
        'storage': 'softraid-1x500nvme',
        'bandwidth': 'bw-300-mbps',
        'vrack': 'none',
        'fqn': '24ska01.ram-4g.softraid-1x500nvme.gra',
        'price': 10.0,
        'fee': 0.0,
    }
    base.update(kw)
    return base


def test_sort_by_price_ascending_and_descending():
    plans = [_plan(price=20.0), _plan(price=8.0), _plan(price=12.0)]
    assert [p['price'] for p in sort_plans(plans, 'price')] == [8.0, 12.0, 20.0]
    assert [p['price'] for p in sort_plans(plans, 'price', reverse=True)] \
        == [20.0, 12.0, 8.0]


def test_price_sorts_numerically_not_as_displayed_string():
    # '9.99' > '10.00' as strings; the cheapest server must still come first.
    plans = [_plan(price=10.0), _plan(price=9.99)]
    assert [p['price'] for p in sort_plans(plans, 'price')] == [9.99, 10.0]


def test_total_sorts_on_price_plus_fee():
    # Cheaper monthly, dearer once the setup fee is counted.
    plans = [_plan(price=9.0, fee=50.0), _plan(price=10.0, fee=0.0)]
    assert [p['price'] for p in sort_plans(plans, 'total')] == [10.0, 9.0]
    assert [p['price'] for p in sort_plans(plans, 'price')] == [9.0, 10.0]


def test_text_column_sorts_naturally():
    # Displayed memory is '4g' / '8g' / '32g'; plain string order would put
    # 32g first.
    plans = [_plan(memory='ram-8g-ddr4'),
             _plan(memory='ram-32g-ddr4'),
             _plan(memory='ram-4g-ddr4')]
    assert [p['memory'] for p in sort_plans(plans, 'memory')] == [
        'ram-4g-ddr4', 'ram-8g-ddr4', 'ram-32g-ddr4']


def test_text_column_sort_is_case_insensitive():
    plans = [_plan(model='ks-b'), _plan(model='KS-A')]
    assert [p['model'] for p in sort_plans(plans, 'model')] == ['KS-A', 'ks-b']


def test_no_key_or_unknown_key_keeps_catalog_order():
    plans = [_plan(price=20.0), _plan(price=8.0)]
    for key in ('', None, 'availability'):
        assert sort_plans(plans, key) == plans


def test_sort_does_not_mutate_the_input():
    plans = [_plan(price=20.0), _plan(price=8.0)]
    before = list(plans)
    sort_plans(plans, 'price')
    assert plans == before


@pytest.mark.parametrize('key', COLUMN_KEYS)
def test_every_column_key_is_sortable(key):
    # The filter bar offers every COLUMN_KEYS entry, and so does sort mode —
    # none of them may blow up on a well-formed plan.
    plans = [_plan(), _plan(datacenter='rbx', price=5.0, fee=1.0)]
    assert len(sort_plans(plans, key)) == 2


# ---------------- Header marker (m.interactive) ----------------

def _state(**kw):
    from m.conf import BuyOvhConfig
    cfg = BuyOvhConfig(showFqn=False, showCpu=False, showBandwidth=False,
                       showPrice=True, showFee=False, showTotalPrice=False)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _headers(state):
    from m.interactive import _visible_columns
    return [h for h, _j, _k in _visible_columns(state)]


def test_sorted_column_header_carries_a_direction_arrow():
    # The arrow is the whole sort indicator — and, in sort mode, the focus
    # indicator too. Exactly one column may wear it.
    assert '€/mo ▲' in _headers(_state(sortKey='price'))
    assert '€/mo ▼' in _headers(_state(sortKey='price', sortReverse=True))
    marked = [h for h in _headers(_state(sortKey='price')) if '▲' in h]
    assert marked == ['€/mo ▲']


def test_unsorted_headers_are_untouched():
    assert _headers(_state()) == ['#', 'Plan', 'Model', 'DC', 'Mem',
                                  'Storage', '€/mo']
