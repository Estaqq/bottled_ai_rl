from typing import List
from venv import create

from presentation_config import presentation_mode, p_delay

from rs.calculator.enums.card_id import CardId
from rs.calculator.enums.potion_id import PotionId
from rs.calculator.enums.relic_id import RelicId

from rs.game.screen_type import ScreenType
from rs.machine.command import Command
from rs.machine.handlers.handler import Handler
from rs.machine.handlers.handler_action import HandlerAction
from rs.machine.state import GameState

import torch
from torch import nn
from torch import optim
from torch import tensor
from torch.nn import functional as F
import os.path


model_path = "./cardRewardNnModel.pt"

#number of hidden neurons for nn (hopefully used to keep track of strategies)
hidden_size = 10

class CommonCardRewardVPGHandler(Handler):
    def __init__(self):
        self.nn = self.createNN()
        if os.path.isfile(model_path):
            self.nn.load_state_dict(torch.load(model_path, weights_only=False))

    class card_nn(nn.Module):
        def __init__(self, in_size, hidden_size, out_size):
            super().__init__()
            self.hiddenLayer = nn.Linear(in_size, hidden_size)
            self.outLayer = nn.Linear(in_size + hidden_size, out_size)

        def forward(self, input):
            hidden = self.hiddenLayer(input)
            hidden = F.relu(hidden)
            output = self.outLayer(torch.cat(input, output))
            output = F.softmax(hidden)
            return output

    def createNN(self): 
        #Cards + upgraded + Relics +  +Floor + HP + Max HP + Missing HP
        #do not care about potions for now
        in_size =  2*len(CardId) + len(RelicId) + 4
        out_size = len(CardId) + 1
        nn = self.card_nn(in_size, hidden_size, out_size)
        return nn
    
    def buildInput(self, state: GameState)

    def can_handle(self, state: GameState) -> bool:
        return state.has_command(Command.CHOOSE) \
               and state.screen_type() == ScreenType.CARD_REWARD.value \
               and (state.game_state()["room_phase"] == "COMPLETE" or state.game_state()["room_phase"] == "EVENT" or
                    state.game_state()["room_phase"] == "COMBAT")

    def handle(self, state: GameState) -> HandlerAction:
        return HandlerAction(commands=["choose 0", "wait 30"])
        


