from typing import Callable

from rs.calculator.interfaces.battle_state_interface import BattleStateInterface
from rs.calculator.interfaces.card_effects_interface import CardEffectsInterface
from rs.calculator.interfaces.card_interface import CardInterface

CardEffectCustomHook = Callable[[BattleStateInterface, CardEffectsInterface, int], None]
CardEffectCustomHookWithCard = Callable[[CardInterface], None]
