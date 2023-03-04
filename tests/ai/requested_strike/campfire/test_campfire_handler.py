from ai.requested_strike.rs_test_handler_fixture import RsTestHandlerFixture
from rs.ai.requested_strike.handlers.campfire_handler import CampfireHandler


class CampfireHandlerTestCase(RsTestHandlerFixture):
    handler = CampfireHandler

    def test_rest(self):
        self.execute_handler_tests('/campfire/campfire_rest.json', ['choose 0'])

    def test_smith(self):
        self.execute_handler_tests('/campfire/campfire_smith.json', ['choose 1'])

    def test_rest_pantograph_boss(self):
        self.execute_handler_tests('/campfire/campfire_rest_pantograph_boss.json', ['choose 0'])

    def test_rest_pantograph_not_boss(self):
        self.execute_handler_tests('/campfire/campfire_rest_pantograph_not_boss.json', ['choose 0'])

    def test_smith_pantograph_boss(self):
        self.execute_handler_tests('/campfire/campfire_smith_pantograph_boss.json', ['choose 1'])

    def test_smith_pantograph_not_boss(self):
        self.execute_handler_tests('/campfire/campfire_smith_pantograph_not_boss.json', ['choose 1'])