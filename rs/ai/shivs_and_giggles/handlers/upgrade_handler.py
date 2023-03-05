from typing import List

from config import presentation_mode, p_delay, p_delay_s
from rs.machine.command import Command
from rs.machine.handlers.handler import Handler
from rs.machine.state import GameState


class UpgradeHandler(Handler):

    def __init__(self):
        self.priorities: List[str] = [
            'neutralize',
            'a thousand cuts',
            'tools of the trade',
            'accuracy',
            'adrenaline',
            'storm of steel',
            'blade dance',
            'terror',
            'infinite blades',
            'cloak and dagger',
            'die die die',
            'after image',
            'leg sweep',
            'poisoned stab',
            'sucker punch',
            'escape plan',
            'dagger spray',
            'caltrops',
            'heel hook',
            'survivor',
        ]

    def can_handle(self, state: GameState) -> bool:
        return state.has_command(Command.CHOOSE) \
               and state.game_state()["screen_type"] == "GRID" \
               and state.game_state()["screen_state"]["for_upgrade"]

    def handle(self, state: GameState) -> List[str]:
        choice_list = state.game_state()["choice_list"]

        for priority in self.priorities:
            if priority in choice_list:
                if presentation_mode:
                    return [p_delay, "choose " + priority, p_delay_s]
                return ["choose " + priority]
        if presentation_mode:
            return [p_delay, "choose " + choice_list[0], p_delay_s]
        return ["choose " + choice_list[0]]
