"""Cart-building tests.

m.api.build_cart is the one place that talks to the order API, so it is
never exercised by a real run in tests. Here the ovh client is replaced by
a recorder and we assert on the calls that would have been made — mainly
that every mandatory addon of the plan is added as an option, since OVH
rejects a cart that is missing one.
"""
import pytest

import m.api


class _FakeClient:
    """Records posts; answers the two calls whose result build_cart uses."""

    def __init__(self):
        self.posts = []

    def post(self, path, **kwargs):
        self.posts.append((path, kwargs))
        if path == '/order/cart':
            return {'cartId': 'CART1'}
        if path.endswith('/eco'):
            return {'itemId': 42}
        return {}

    def options(self):
        return [kw['planCode'] for path, kw in self.posts
                if path.endswith('/eco/options')]

    def configuration(self):
        return {kw['label']: kw['value'] for path, kw in self.posts
                if path.endswith('/configuration')}


@pytest.fixture
def client(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(m.api, 'client', fake)
    return fake


def _plan(**overrides):
    plan = {
        'planCode': '24sk40',
        'model': 'KS-4',
        'datacenter': 'gra',
        'memory': 'ram-32g-ecc',
        'storage': 'softraid-2x480ssd',
        'bandwidth': 'bandwidth-500',
        'vrack': 'none',
        'options': ['ram-32g-ecc', 'softraid-2x480ssd', 'bandwidth-500'],
    }
    plan.update(overrides)
    return plan


class TestCartOptions:

    def test_options_list_is_used_as_is(self):
        plan = _plan(options=['ram-64g', 'softraid-2x960nvme',
                              'bandwidth-1000', 'gpu-1xradeon-rx6700xt-12g'])
        assert m.api.cart_options(plan) == [
            'ram-64g', 'softraid-2x960nvme', 'bandwidth-1000',
            'gpu-1xradeon-rx6700xt-12g']

    def test_legacy_row_without_options_falls_back(self):
        plan = _plan()
        del plan['options']
        assert m.api.cart_options(plan) == ['ram-32g-ecc',
                                            'softraid-2x480ssd',
                                            'bandwidth-500']

    def test_legacy_row_keeps_a_real_vrack(self):
        plan = _plan(vrack='vrack-bandwidth-100')
        del plan['options']
        assert m.api.cart_options(plan)[-1] == 'vrack-bandwidth-100'

    def test_legacy_row_drops_vrack_none(self):
        plan = _plan()
        del plan['options']
        assert 'none' not in m.api.cart_options(plan)


class TestBuildCart:

    def test_gpu_option_is_posted(self, client):
        gpu = 'gpu-1xradeon-rx6700xt-12g-26risegpu01-v1'
        plan = _plan(planCode='26risegpu01-v1',
                     options=['ram-64g', 'softraid-2x960nvme',
                              'bandwidth-1000', gpu])
        m.api.build_cart(plan, 'FR', False, 1)
        assert client.options() == ['ram-64g', 'softraid-2x960nvme',
                                    'bandwidth-1000', gpu]

    def test_server_and_options_share_the_item_and_mode(self, client):
        m.api.build_cart(_plan(), 'FR', False, 24)
        eco = [kw for path, kw in client.posts if path.endswith('/eco')][0]
        assert (eco['planCode'], eco['duration'],
                eco['pricingMode']) == ('24sk40', 'P24M', 'upfront24')
        for path, kw in client.posts:
            if path.endswith('/eco/options'):
                assert kw['itemId'] == 42
                assert kw['duration'] == 'P24M'
                assert kw['pricingMode'] == 'upfront24'

    def test_datacenter_and_region_configuration(self, client):
        m.api.build_cart(_plan(datacenter='sgp'), 'FR', False, 1)
        assert client.configuration() == {'dedicated_datacenter': 'sgp',
                                          'dedicated_os': 'none_64.en',
                                          'region': 'canada'}

    def test_fake_buy_posts_nothing(self, client):
        assert m.api.build_cart(_plan(), 'FR', True, 1) == 0
        assert client.posts == []

    def test_not_logged_in_raises(self, monkeypatch):
        monkeypatch.setattr(m.api, 'client', None)
        with pytest.raises(m.api.NotLoggedIn):
            m.api.build_cart(_plan(), 'FR', False, 1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
