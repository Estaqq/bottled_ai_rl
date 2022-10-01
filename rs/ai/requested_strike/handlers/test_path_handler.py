import json
import unittest

from rs.ai.requested_strike.handlers.path_handler import PathHandler
from rs.machine.state import GameState

STATE = '{"available_commands":["choose","return","key","click","wait","state"],"ready_for_command":true,"in_game":true,"game_state":{"choice_list":["x=0"],"screen_type":"MAP","screen_state":{"first_node_chosen":true,"current_node":{"symbol":"M","x":0,"y":0},"boss_available":false,"next_nodes":[{"symbol":"M","x":0,"y":1}]},"seed":7230305506610474000,"deck":[{"exhausts":false,"is_playable":false,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"99d097df-4e3d-43d1-8bac-0ba8c07bf9f5","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"is_playable":false,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"543cbb0c-12c8-4f1a-be61-9274011c0c6e","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"is_playable":false,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"158a2b9b-2db1-429a-812d-0ca6945a0091","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"is_playable":false,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"bd884320-0bb9-48c4-8228-317197bd11be","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"is_playable":false,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"4252b0b5-69ee-4353-ba8f-3391b07c79ce","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"is_playable":false,"cost":1,"name":"Defend","id":"Defend_R","type":"SKILL","ethereal":false,"uuid":"05641bb6-a96b-415a-be1d-a2f5861f6d08","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"is_playable":false,"cost":1,"name":"Defend","id":"Defend_R","type":"SKILL","ethereal":false,"uuid":"c46eba10-694a-4698-af0f-79902b3ee782","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"is_playable":false,"cost":1,"name":"Defend","id":"Defend_R","type":"SKILL","ethereal":false,"uuid":"b97d1ee7-15fb-4d38-9b1e-68d4affc3491","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"is_playable":false,"cost":1,"name":"Defend","id":"Defend_R","type":"SKILL","ethereal":false,"uuid":"fbc25ad6-c6e8-4fec-b21a-96b1a151d193","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"is_playable":false,"cost":2,"name":"Bash+","id":"Bash","type":"ATTACK","ethereal":false,"uuid":"34485847-0c88-4365-97ac-1001d904a338","upgrades":1,"rarity":"BASIC","has_target":true},{"exhausts":true,"is_playable":false,"cost":2,"name":"Shockwave","id":"Shockwave","type":"SKILL","ethereal":false,"uuid":"0f968c81-10a0-478b-9dbd-149f9d043329","upgrades":0,"rarity":"UNCOMMON","has_target":false}],"relics":[{"name":"Burning Blood","id":"Burning Blood","counter":-1}],"max_hp":80,"act_boss":"Slime Boss","gold":111,"action_phase":"WAITING_ON_USER","act":1,"screen_name":"MAP","room_phase":"COMPLETE","is_screen_up":true,"potions":[{"requires_target":false,"can_use":false,"can_discard":false,"name":"Potion Slot","id":"Potion Slot"},{"requires_target":false,"can_use":false,"can_discard":false,"name":"Potion Slot","id":"Potion Slot"},{"requires_target":false,"can_use":false,"can_discard":false,"name":"Potion Slot","id":"Potion Slot"}],"current_hp":80,"floor":1,"ascension_level":0,"class":"IRONCLAD","map":[{"symbol":"M","children":[{"x":0,"y":1}],"x":0,"y":0,"parents":[]},{"symbol":"M","children":[{"x":2,"y":1}],"x":1,"y":0,"parents":[]},{"symbol":"M","children":[{"x":3,"y":1}],"x":3,"y":0,"parents":[]},{"symbol":"M","children":[{"x":4,"y":1}],"x":4,"y":0,"parents":[]},{"symbol":"M","children":[{"x":6,"y":1}],"x":6,"y":0,"parents":[]},{"symbol":"M","children":[{"x":1,"y":2}],"x":0,"y":1,"parents":[]},{"symbol":"M","children":[{"x":1,"y":2}],"x":2,"y":1,"parents":[]},{"symbol":"M","children":[{"x":2,"y":2},{"x":4,"y":2}],"x":3,"y":1,"parents":[]},{"symbol":"?","children":[{"x":5,"y":2}],"x":4,"y":1,"parents":[]},{"symbol":"M","children":[{"x":6,"y":2}],"x":6,"y":1,"parents":[]},{"symbol":"M","children":[{"x":2,"y":3}],"x":1,"y":2,"parents":[]},{"symbol":"M","children":[{"x":2,"y":3}],"x":2,"y":2,"parents":[]},{"symbol":"?","children":[{"x":4,"y":3}],"x":4,"y":2,"parents":[]},{"symbol":"?","children":[{"x":5,"y":3}],"x":5,"y":2,"parents":[]},{"symbol":"?","children":[{"x":6,"y":3}],"x":6,"y":2,"parents":[]},{"symbol":"M","children":[{"x":3,"y":4}],"x":2,"y":3,"parents":[]},{"symbol":"M","children":[{"x":3,"y":4}],"x":4,"y":3,"parents":[]},{"symbol":"M","children":[{"x":4,"y":4}],"x":5,"y":3,"parents":[]},{"symbol":"M","children":[{"x":5,"y":4}],"x":6,"y":3,"parents":[]},{"symbol":"?","children":[{"x":3,"y":5},{"x":4,"y":5}],"x":3,"y":4,"parents":[]},{"symbol":"M","children":[{"x":4,"y":5}],"x":4,"y":4,"parents":[]},{"symbol":"M","children":[{"x":5,"y":5}],"x":5,"y":4,"parents":[]},{"symbol":"E","children":[{"x":3,"y":6}],"x":3,"y":5,"parents":[]},{"symbol":"R","children":[{"x":3,"y":6},{"x":5,"y":6}],"x":4,"y":5,"parents":[]},{"symbol":"R","children":[{"x":5,"y":6}],"x":5,"y":5,"parents":[]},{"symbol":"M","children":[{"x":2,"y":7},{"x":3,"y":7},{"x":4,"y":7}],"x":3,"y":6,"parents":[]},{"symbol":"E","children":[{"x":6,"y":7}],"x":5,"y":6,"parents":[]},{"symbol":"E","children":[{"x":2,"y":8}],"x":2,"y":7,"parents":[]},{"symbol":"M","children":[{"x":2,"y":8}],"x":3,"y":7,"parents":[]},{"symbol":"$","children":[{"x":5,"y":8}],"x":4,"y":7,"parents":[]},{"symbol":"M","children":[{"x":5,"y":8}],"x":6,"y":7,"parents":[]},{"symbol":"T","children":[{"x":2,"y":9}],"x":2,"y":8,"parents":[]},{"symbol":"T","children":[{"x":5,"y":9},{"x":6,"y":9}],"x":5,"y":8,"parents":[]},{"symbol":"?","children":[{"x":2,"y":10},{"x":3,"y":10}],"x":2,"y":9,"parents":[]},{"symbol":"M","children":[{"x":4,"y":10},{"x":5,"y":10}],"x":5,"y":9,"parents":[]},{"symbol":"?","children":[{"x":5,"y":10}],"x":6,"y":9,"parents":[]},{"symbol":"E","children":[{"x":2,"y":11}],"x":2,"y":10,"parents":[]},{"symbol":"R","children":[{"x":4,"y":11}],"x":3,"y":10,"parents":[]},{"symbol":"?","children":[{"x":4,"y":11},{"x":5,"y":11}],"x":4,"y":10,"parents":[]},{"symbol":"$","children":[{"x":6,"y":11}],"x":5,"y":10,"parents":[]},{"symbol":"M","children":[{"x":1,"y":12}],"x":2,"y":11,"parents":[]},{"symbol":"M","children":[{"x":3,"y":12},{"x":5,"y":12}],"x":4,"y":11,"parents":[]},{"symbol":"?","children":[{"x":5,"y":12}],"x":5,"y":11,"parents":[]},{"symbol":"M","children":[{"x":5,"y":12},{"x":6,"y":12}],"x":6,"y":11,"parents":[]},{"symbol":"$","children":[{"x":1,"y":13}],"x":1,"y":12,"parents":[]},{"symbol":"R","children":[{"x":3,"y":13}],"x":3,"y":12,"parents":[]},{"symbol":"M","children":[{"x":4,"y":13},{"x":5,"y":13}],"x":5,"y":12,"parents":[]},{"symbol":"R","children":[{"x":6,"y":13}],"x":6,"y":12,"parents":[]},{"symbol":"?","children":[{"x":1,"y":14}],"x":1,"y":13,"parents":[]},{"symbol":"?","children":[{"x":3,"y":14}],"x":3,"y":13,"parents":[]},{"symbol":"M","children":[{"x":3,"y":14},{"x":5,"y":14}],"x":4,"y":13,"parents":[]},{"symbol":"M","children":[{"x":5,"y":14}],"x":5,"y":13,"parents":[]},{"symbol":"M","children":[{"x":6,"y":14}],"x":6,"y":13,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":1,"y":14,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":3,"y":14,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":5,"y":14,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":6,"y":14,"parents":[]}],"room_type":"MonsterRoom"}}'
CHOOSE_ELITES = '{"available_commands":["choose","potion","return","key","click","wait","state"],"ready_for_command":true,"in_game":true,"game_state":{"choice_list":["x\u003d2","x\u003d3"],"screen_type":"MAP","screen_state":{"first_node_chosen":true,"current_node":{"symbol":"M","x":2,"y":2},"boss_available":false,"next_nodes":[{"symbol":"$","x":2,"y":3},{"symbol":"?","x":3,"y":3}]},"seed":2283446537348531365,"deck":[{"exhausts":false,"is_playable":false,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"9846e10a-51a8-4e32-9a6a-4df458575c05","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"is_playable":false,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"ceb096ab-1434-467e-928d-08cea46a7c79","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"is_playable":false,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"cfbc8c41-35f8-4c55-8ff1-f05e5f4d42c5","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"is_playable":false,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"5905e0ba-154a-4252-aa6d-30566a33e0db","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"is_playable":false,"cost":1,"name":"Defend","id":"Defend_R","type":"SKILL","ethereal":false,"uuid":"21f72450-0835-4e87-8c2b-cfaf56be58ee","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"is_playable":false,"cost":1,"name":"Defend","id":"Defend_R","type":"SKILL","ethereal":false,"uuid":"493255b6-58ff-43fe-bc5f-9580e2d60480","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"is_playable":false,"cost":1,"name":"Defend","id":"Defend_R","type":"SKILL","ethereal":false,"uuid":"690cf697-836c-4979-abbe-1a4dadd0da08","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"is_playable":false,"cost":1,"name":"Defend","id":"Defend_R","type":"SKILL","ethereal":false,"uuid":"16a7f3e6-d739-40a7-9dd2-9f5b8bab8ab6","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"is_playable":false,"cost":2,"name":"Bash","id":"Bash","type":"ATTACK","ethereal":false,"uuid":"50b3e371-dffb-45f3-b418-76fd57bd843a","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"is_playable":false,"cost":0,"name":"Battle Trance","id":"Battle Trance","type":"SKILL","ethereal":false,"uuid":"14cd602f-6974-40e0-8a24-6a84e941fa11","upgrades":0,"rarity":"UNCOMMON","has_target":false},{"exhausts":false,"is_playable":false,"cost":1,"name":"Thunderclap","id":"Thunderclap","type":"ATTACK","ethereal":false,"uuid":"f509e108-afe2-45c9-85e6-1d5d960bc84a","upgrades":0,"rarity":"COMMON","has_target":false},{"exhausts":false,"is_playable":false,"cost":1,"name":"Twin Strike","id":"Twin Strike","type":"ATTACK","ethereal":false,"uuid":"ec302f83-f64b-4c25-b045-94ce00a2ea81","upgrades":0,"rarity":"COMMON","has_target":true}],"relics":[{"name":"Burning Blood","id":"Burning Blood","counter":-1}],"max_hp":80,"act_boss":"Hexaghost","gold":146,"action_phase":"WAITING_ON_USER","act":1,"screen_name":"MAP","room_phase":"COMPLETE","is_screen_up":true,"potions":[{"requires_target":true,"can_use":false,"can_discard":true,"name":"Fear Potion","id":"FearPotion"},{"requires_target":false,"can_use":false,"can_discard":false,"name":"Potion Slot","id":"Potion Slot"},{"requires_target":false,"can_use":false,"can_discard":false,"name":"Potion Slot","id":"Potion Slot"}],"current_hp":80,"floor":3,"ascension_level":0,"class":"IRONCLAD","map":[{"symbol":"M","children":[{"x":1,"y":1}],"x":0,"y":0,"parents":[]},{"symbol":"M","children":[{"x":3,"y":1},{"x":5,"y":1}],"x":4,"y":0,"parents":[]},{"symbol":"M","children":[{"x":2,"y":2}],"x":1,"y":1,"parents":[]},{"symbol":"M","children":[{"x":3,"y":2}],"x":3,"y":1,"parents":[]},{"symbol":"?","children":[{"x":4,"y":2},{"x":6,"y":2}],"x":5,"y":1,"parents":[]},{"symbol":"M","children":[{"x":2,"y":3},{"x":3,"y":3}],"x":2,"y":2,"parents":[]},{"symbol":"M","children":[{"x":3,"y":3}],"x":3,"y":2,"parents":[]},{"symbol":"M","children":[{"x":3,"y":3},{"x":4,"y":3}],"x":4,"y":2,"parents":[]},{"symbol":"?","children":[{"x":5,"y":3}],"x":6,"y":2,"parents":[]},{"symbol":"$","children":[{"x":3,"y":4}],"x":2,"y":3,"parents":[]},{"symbol":"?","children":[{"x":3,"y":4},{"x":4,"y":4}],"x":3,"y":3,"parents":[]},{"symbol":"M","children":[{"x":5,"y":4}],"x":4,"y":3,"parents":[]},{"symbol":"?","children":[{"x":5,"y":4}],"x":5,"y":3,"parents":[]},{"symbol":"M","children":[{"x":2,"y":5}],"x":3,"y":4,"parents":[]},{"symbol":"?","children":[{"x":3,"y":5},{"x":4,"y":5}],"x":4,"y":4,"parents":[]},{"symbol":"M","children":[{"x":4,"y":5},{"x":6,"y":5}],"x":5,"y":4,"parents":[]},{"symbol":"R","children":[{"x":1,"y":6},{"x":2,"y":6}],"x":2,"y":5,"parents":[]},{"symbol":"R","children":[{"x":2,"y":6}],"x":3,"y":5,"parents":[]},{"symbol":"M","children":[{"x":3,"y":6},{"x":4,"y":6}],"x":4,"y":5,"parents":[]},{"symbol":"R","children":[{"x":5,"y":6}],"x":6,"y":5,"parents":[]},{"symbol":"M","children":[{"x":1,"y":7}],"x":1,"y":6,"parents":[]},{"symbol":"?","children":[{"x":3,"y":7}],"x":2,"y":6,"parents":[]},{"symbol":"E","children":[{"x":3,"y":7}],"x":3,"y":6,"parents":[]},{"symbol":"M","children":[{"x":5,"y":7}],"x":4,"y":6,"parents":[]},{"symbol":"?","children":[{"x":5,"y":7}],"x":5,"y":6,"parents":[]},{"symbol":"M","children":[{"x":2,"y":8}],"x":1,"y":7,"parents":[]},{"symbol":"M","children":[{"x":2,"y":8},{"x":4,"y":8}],"x":3,"y":7,"parents":[]},{"symbol":"E","children":[{"x":5,"y":8},{"x":6,"y":8}],"x":5,"y":7,"parents":[]},{"symbol":"T","children":[{"x":1,"y":9},{"x":3,"y":9}],"x":2,"y":8,"parents":[]},{"symbol":"T","children":[{"x":5,"y":9}],"x":4,"y":8,"parents":[]},{"symbol":"T","children":[{"x":5,"y":9}],"x":5,"y":8,"parents":[]},{"symbol":"T","children":[{"x":6,"y":9}],"x":6,"y":8,"parents":[]},{"symbol":"?","children":[{"x":2,"y":10}],"x":1,"y":9,"parents":[]},{"symbol":"R","children":[{"x":3,"y":10},{"x":4,"y":10}],"x":3,"y":9,"parents":[]},{"symbol":"?","children":[{"x":4,"y":10}],"x":5,"y":9,"parents":[]},{"symbol":"$","children":[{"x":5,"y":10}],"x":6,"y":9,"parents":[]},{"symbol":"M","children":[{"x":3,"y":11}],"x":2,"y":10,"parents":[]},{"symbol":"?","children":[{"x":3,"y":11}],"x":3,"y":10,"parents":[]},{"symbol":"M","children":[{"x":5,"y":11}],"x":4,"y":10,"parents":[]},{"symbol":"M","children":[{"x":6,"y":11}],"x":5,"y":10,"parents":[]},{"symbol":"M","children":[{"x":3,"y":12},{"x":4,"y":12}],"x":3,"y":11,"parents":[]},{"symbol":"R","children":[{"x":4,"y":12},{"x":6,"y":12}],"x":5,"y":11,"parents":[]},{"symbol":"R","children":[{"x":6,"y":12}],"x":6,"y":11,"parents":[]},{"symbol":"E","children":[{"x":3,"y":13}],"x":3,"y":12,"parents":[]},{"symbol":"M","children":[{"x":4,"y":13},{"x":5,"y":13}],"x":4,"y":12,"parents":[]},{"symbol":"E","children":[{"x":5,"y":13},{"x":6,"y":13}],"x":6,"y":12,"parents":[]},{"symbol":"M","children":[{"x":4,"y":14}],"x":3,"y":13,"parents":[]},{"symbol":"?","children":[{"x":4,"y":14}],"x":4,"y":13,"parents":[]},{"symbol":"M","children":[{"x":4,"y":14},{"x":5,"y":14}],"x":5,"y":13,"parents":[]},{"symbol":"M","children":[{"x":5,"y":14},{"x":6,"y":14}],"x":6,"y":13,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":4,"y":14,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":5,"y":14,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":6,"y":14,"parents":[]}],"room_type":"MonsterRoom"}}'
INITIAL_STATE = '{"available_commands":["choose","return","key","click","wait","state"],"ready_for_command":true,"in_game":true,"game_state":{"choice_list":["x\u003d0","x\u003d4"],"screen_type":"MAP","screen_state":{"first_node_chosen":false,"current_node":{"x":0,"y":-1},"boss_available":false,"next_nodes":[{"symbol":"M","x":0,"y":0},{"symbol":"M","x":4,"y":0}]},"seed":2283446537348531365,"deck":[{"exhausts":false,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"4a01aedb-65f9-44f6-90bd-9f66c1fc1cdd","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"c41ad7e1-ae42-461b-90ec-ca948a74813a","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"f84f97da-5fb8-4c11-88fa-abc343685610","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"855622a3-0773-4e89-b50c-d48fcba44b0f","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"cost":1,"name":"Defend","id":"Defend_R","type":"SKILL","ethereal":false,"uuid":"5c43df3d-6da0-46b7-811d-1a192b9762f5","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"cost":1,"name":"Defend","id":"Defend_R","type":"SKILL","ethereal":false,"uuid":"16e758cd-2220-4b64-9baa-75036da50e69","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"cost":1,"name":"Defend","id":"Defend_R","type":"SKILL","ethereal":false,"uuid":"90e0a3be-d1b5-4d92-b77a-48c7baed64e6","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"cost":1,"name":"Defend","id":"Defend_R","type":"SKILL","ethereal":false,"uuid":"c6ec2b6b-5eb3-488a-b5ff-0f4ddb64ada4","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"cost":2,"name":"Bash","id":"Bash","type":"ATTACK","ethereal":false,"uuid":"bd0add78-5df7-4e92-8d1f-15c3247d6d33","upgrades":0,"rarity":"BASIC","has_target":true}],"relics":[{"name":"Burning Blood","id":"Burning Blood","counter":-1}],"max_hp":80,"act_boss":"Hexaghost","gold":99,"action_phase":"WAITING_ON_USER","act":1,"screen_name":"MAP","room_phase":"COMPLETE","is_screen_up":true,"potions":[{"requires_target":false,"can_use":false,"can_discard":false,"name":"Potion Slot","id":"Potion Slot"},{"requires_target":false,"can_use":false,"can_discard":false,"name":"Potion Slot","id":"Potion Slot"},{"requires_target":false,"can_use":false,"can_discard":false,"name":"Potion Slot","id":"Potion Slot"}],"current_hp":80,"floor":0,"ascension_level":0,"class":"IRONCLAD","map":[{"symbol":"M","children":[{"x":1,"y":1}],"x":0,"y":0,"parents":[]},{"symbol":"M","children":[{"x":3,"y":1},{"x":5,"y":1}],"x":4,"y":0,"parents":[]},{"symbol":"M","children":[{"x":2,"y":2}],"x":1,"y":1,"parents":[]},{"symbol":"M","children":[{"x":3,"y":2}],"x":3,"y":1,"parents":[]},{"symbol":"?","children":[{"x":4,"y":2},{"x":6,"y":2}],"x":5,"y":1,"parents":[]},{"symbol":"M","children":[{"x":2,"y":3},{"x":3,"y":3}],"x":2,"y":2,"parents":[]},{"symbol":"M","children":[{"x":3,"y":3}],"x":3,"y":2,"parents":[]},{"symbol":"M","children":[{"x":3,"y":3},{"x":4,"y":3}],"x":4,"y":2,"parents":[]},{"symbol":"?","children":[{"x":5,"y":3}],"x":6,"y":2,"parents":[]},{"symbol":"$","children":[{"x":3,"y":4}],"x":2,"y":3,"parents":[]},{"symbol":"?","children":[{"x":3,"y":4},{"x":4,"y":4}],"x":3,"y":3,"parents":[]},{"symbol":"M","children":[{"x":5,"y":4}],"x":4,"y":3,"parents":[]},{"symbol":"?","children":[{"x":5,"y":4}],"x":5,"y":3,"parents":[]},{"symbol":"M","children":[{"x":2,"y":5}],"x":3,"y":4,"parents":[]},{"symbol":"?","children":[{"x":3,"y":5},{"x":4,"y":5}],"x":4,"y":4,"parents":[]},{"symbol":"M","children":[{"x":4,"y":5},{"x":6,"y":5}],"x":5,"y":4,"parents":[]},{"symbol":"R","children":[{"x":1,"y":6},{"x":2,"y":6}],"x":2,"y":5,"parents":[]},{"symbol":"R","children":[{"x":2,"y":6}],"x":3,"y":5,"parents":[]},{"symbol":"M","children":[{"x":3,"y":6},{"x":4,"y":6}],"x":4,"y":5,"parents":[]},{"symbol":"R","children":[{"x":5,"y":6}],"x":6,"y":5,"parents":[]},{"symbol":"M","children":[{"x":1,"y":7}],"x":1,"y":6,"parents":[]},{"symbol":"?","children":[{"x":3,"y":7}],"x":2,"y":6,"parents":[]},{"symbol":"E","children":[{"x":3,"y":7}],"x":3,"y":6,"parents":[]},{"symbol":"M","children":[{"x":5,"y":7}],"x":4,"y":6,"parents":[]},{"symbol":"?","children":[{"x":5,"y":7}],"x":5,"y":6,"parents":[]},{"symbol":"M","children":[{"x":2,"y":8}],"x":1,"y":7,"parents":[]},{"symbol":"M","children":[{"x":2,"y":8},{"x":4,"y":8}],"x":3,"y":7,"parents":[]},{"symbol":"E","children":[{"x":5,"y":8},{"x":6,"y":8}],"x":5,"y":7,"parents":[]},{"symbol":"T","children":[{"x":1,"y":9},{"x":3,"y":9}],"x":2,"y":8,"parents":[]},{"symbol":"T","children":[{"x":5,"y":9}],"x":4,"y":8,"parents":[]},{"symbol":"T","children":[{"x":5,"y":9}],"x":5,"y":8,"parents":[]},{"symbol":"T","children":[{"x":6,"y":9}],"x":6,"y":8,"parents":[]},{"symbol":"?","children":[{"x":2,"y":10}],"x":1,"y":9,"parents":[]},{"symbol":"R","children":[{"x":3,"y":10},{"x":4,"y":10}],"x":3,"y":9,"parents":[]},{"symbol":"?","children":[{"x":4,"y":10}],"x":5,"y":9,"parents":[]},{"symbol":"$","children":[{"x":5,"y":10}],"x":6,"y":9,"parents":[]},{"symbol":"M","children":[{"x":3,"y":11}],"x":2,"y":10,"parents":[]},{"symbol":"?","children":[{"x":3,"y":11}],"x":3,"y":10,"parents":[]},{"symbol":"M","children":[{"x":5,"y":11}],"x":4,"y":10,"parents":[]},{"symbol":"M","children":[{"x":6,"y":11}],"x":5,"y":10,"parents":[]},{"symbol":"M","children":[{"x":3,"y":12},{"x":4,"y":12}],"x":3,"y":11,"parents":[]},{"symbol":"R","children":[{"x":4,"y":12},{"x":6,"y":12}],"x":5,"y":11,"parents":[]},{"symbol":"R","children":[{"x":6,"y":12}],"x":6,"y":11,"parents":[]},{"symbol":"E","children":[{"x":3,"y":13}],"x":3,"y":12,"parents":[]},{"symbol":"M","children":[{"x":4,"y":13},{"x":5,"y":13}],"x":4,"y":12,"parents":[]},{"symbol":"E","children":[{"x":5,"y":13},{"x":6,"y":13}],"x":6,"y":12,"parents":[]},{"symbol":"M","children":[{"x":4,"y":14}],"x":3,"y":13,"parents":[]},{"symbol":"?","children":[{"x":4,"y":14}],"x":4,"y":13,"parents":[]},{"symbol":"M","children":[{"x":4,"y":14},{"x":5,"y":14}],"x":5,"y":13,"parents":[]},{"symbol":"M","children":[{"x":5,"y":14},{"x":6,"y":14}],"x":6,"y":13,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":4,"y":14,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":5,"y":14,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":6,"y":14,"parents":[]}],"room_type":"NeowRoom"}}'
ACT_2_START = '{"available_commands":["choose","potion","key","click","wait","state"],"ready_for_command":true,"in_game":true,"game_state":{"choice_list":["x\u003d0","x\u003d2","x\u003d4","x\u003d5"],"screen_type":"MAP","screen_state":{"first_node_chosen":false,"current_node":{"x":0,"y":-1},"boss_available":false,"next_nodes":[{"symbol":"M","x":0,"y":0},{"symbol":"M","x":2,"y":0},{"symbol":"M","x":4,"y":0},{"symbol":"M","x":5,"y":0}]},"seed":2283446537348531365,"deck":[{"exhausts":false,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"e0a1d82f-668c-4156-b5cd-b6bfc806cdac","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"32c8e211-4736-4cc7-ab2d-9f643a09f76a","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"c0b2fc32-d768-4631-b29c-fd40b2aa2b42","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"8eff8536-354e-4d96-9c2e-886b4438734d","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"c31791ff-f86b-4342-99e2-fe9408216bc9","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"cost":1,"name":"Defend","id":"Defend_R","type":"SKILL","ethereal":false,"uuid":"8b2612e9-06fb-4bc5-8fd1-f4404821876b","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"cost":1,"name":"Defend","id":"Defend_R","type":"SKILL","ethereal":false,"uuid":"b6e23dc8-1225-4506-b5c1-bbb9eab2b423","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"cost":2,"name":"Bash","id":"Bash","type":"ATTACK","ethereal":false,"uuid":"e15755c4-a98d-47da-b30c-1516b9404f81","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":true,"cost":1,"name":"Pummel","id":"Pummel","type":"ATTACK","ethereal":false,"uuid":"d17e724f-768d-453c-9984-16bdc34f113b","upgrades":0,"rarity":"UNCOMMON","has_target":true},{"exhausts":false,"cost":1,"name":"Thunderclap","id":"Thunderclap","type":"ATTACK","ethereal":false,"uuid":"024a68d0-3994-4829-8058-9ce1c33bf080","upgrades":0,"rarity":"COMMON","has_target":false},{"exhausts":true,"cost":2,"name":"Shockwave","id":"Shockwave","type":"SKILL","ethereal":false,"uuid":"3f7948dc-e986-4349-b3b8-bf8f51687770","upgrades":0,"rarity":"UNCOMMON","has_target":false},{"exhausts":false,"cost":2,"name":"Perfected Strike","id":"Perfected Strike","type":"ATTACK","ethereal":false,"uuid":"cdafc1c4-9c8d-44e2-a120-37c39083e331","upgrades":0,"rarity":"COMMON","has_target":true},{"exhausts":false,"cost":1,"name":"Pommel Strike","id":"Pommel Strike","type":"ATTACK","ethereal":false,"uuid":"8fd5f434-2834-4a7f-a452-212a1fcc8071","upgrades":0,"rarity":"COMMON","has_target":true},{"exhausts":false,"cost":1,"name":"Pommel Strike","id":"Pommel Strike","type":"ATTACK","ethereal":false,"uuid":"2ee4fff4-a1a3-4019-9829-be13526285a5","upgrades":0,"rarity":"COMMON","has_target":true},{"exhausts":false,"cost":1,"name":"Twin Strike","id":"Twin Strike","type":"ATTACK","ethereal":false,"uuid":"c68baed1-846d-4eda-a1c2-35f83fe8ca86","upgrades":0,"rarity":"COMMON","has_target":true},{"exhausts":true,"cost":2,"name":"Shockwave","id":"Shockwave","type":"SKILL","ethereal":false,"uuid":"d49abb15-2542-4791-861f-cc90b9834b84","upgrades":0,"rarity":"UNCOMMON","has_target":false}],"relics":[{"name":"Burning Blood","id":"Burning Blood","counter":-1},{"name":"Strike Dummy","id":"StrikeDummy","counter":-1},{"name":"Blue Candle","id":"Blue Candle","counter":-1},{"name":"Paper Phrog","id":"Paper Frog","counter":-1},{"name":"Sozu","id":"Sozu","counter":-1}],"max_hp":80,"act_boss":"Collector","gold":379,"action_phase":"WAITING_ON_USER","act":2,"screen_name":"MAP","room_phase":"COMPLETE","is_screen_up":true,"potions":[{"requires_target":false,"can_use":true,"can_discard":true,"name":"Entropic Brew","id":"EntropicBrew"},{"requires_target":false,"can_use":false,"can_discard":true,"name":"Attack Potion","id":"AttackPotion"},{"requires_target":false,"can_use":false,"can_discard":true,"name":"Gambler\u0027s Brew","id":"GamblersBrew"}],"current_hp":80,"floor":17,"ascension_level":0,"class":"IRONCLAD","map":[{"symbol":"M","children":[{"x":0,"y":1},{"x":1,"y":1}],"x":0,"y":0,"parents":[]},{"symbol":"M","children":[{"x":2,"y":1}],"x":2,"y":0,"parents":[]},{"symbol":"M","children":[{"x":3,"y":1}],"x":4,"y":0,"parents":[]},{"symbol":"M","children":[{"x":4,"y":1}],"x":5,"y":0,"parents":[]},{"symbol":"?","children":[{"x":1,"y":2}],"x":0,"y":1,"parents":[]},{"symbol":"M","children":[{"x":1,"y":2},{"x":2,"y":2}],"x":1,"y":1,"parents":[]},{"symbol":"?","children":[{"x":2,"y":2}],"x":2,"y":1,"parents":[]},{"symbol":"?","children":[{"x":2,"y":2}],"x":3,"y":1,"parents":[]},{"symbol":"M","children":[{"x":5,"y":2}],"x":4,"y":1,"parents":[]},{"symbol":"M","children":[{"x":0,"y":3},{"x":1,"y":3}],"x":1,"y":2,"parents":[]},{"symbol":"?","children":[{"x":2,"y":3},{"x":3,"y":3}],"x":2,"y":2,"parents":[]},{"symbol":"M","children":[{"x":5,"y":3}],"x":5,"y":2,"parents":[]},{"symbol":"M","children":[{"x":1,"y":4}],"x":0,"y":3,"parents":[]},{"symbol":"?","children":[{"x":2,"y":4}],"x":1,"y":3,"parents":[]},{"symbol":"M","children":[{"x":2,"y":4}],"x":2,"y":3,"parents":[]},{"symbol":"?","children":[{"x":2,"y":4},{"x":3,"y":4}],"x":3,"y":3,"parents":[]},{"symbol":"?","children":[{"x":6,"y":4}],"x":5,"y":3,"parents":[]},{"symbol":"$","children":[{"x":2,"y":5}],"x":1,"y":4,"parents":[]},{"symbol":"M","children":[{"x":2,"y":5}],"x":2,"y":4,"parents":[]},{"symbol":"?","children":[{"x":3,"y":5}],"x":3,"y":4,"parents":[]},{"symbol":"M","children":[{"x":6,"y":5}],"x":6,"y":4,"parents":[]},{"symbol":"E","children":[{"x":1,"y":6},{"x":2,"y":6},{"x":3,"y":6}],"x":2,"y":5,"parents":[]},{"symbol":"R","children":[{"x":4,"y":6}],"x":3,"y":5,"parents":[]},{"symbol":"E","children":[{"x":6,"y":6}],"x":6,"y":5,"parents":[]},{"symbol":"R","children":[{"x":0,"y":7},{"x":2,"y":7}],"x":1,"y":6,"parents":[]},{"symbol":"$","children":[{"x":2,"y":7}],"x":2,"y":6,"parents":[]},{"symbol":"?","children":[{"x":3,"y":7}],"x":3,"y":6,"parents":[]},{"symbol":"E","children":[{"x":3,"y":7}],"x":4,"y":6,"parents":[]},{"symbol":"?","children":[{"x":5,"y":7}],"x":6,"y":6,"parents":[]},{"symbol":"$","children":[{"x":0,"y":8}],"x":0,"y":7,"parents":[]},{"symbol":"M","children":[{"x":2,"y":8}],"x":2,"y":7,"parents":[]},{"symbol":"R","children":[{"x":2,"y":8},{"x":3,"y":8}],"x":3,"y":7,"parents":[]},{"symbol":"E","children":[{"x":5,"y":8}],"x":5,"y":7,"parents":[]},{"symbol":"T","children":[{"x":0,"y":9}],"x":0,"y":8,"parents":[]},{"symbol":"T","children":[{"x":2,"y":9},{"x":3,"y":9}],"x":2,"y":8,"parents":[]},{"symbol":"T","children":[{"x":3,"y":9}],"x":3,"y":8,"parents":[]},{"symbol":"T","children":[{"x":4,"y":9}],"x":5,"y":8,"parents":[]},{"symbol":"M","children":[{"x":1,"y":10}],"x":0,"y":9,"parents":[]},{"symbol":"M","children":[{"x":3,"y":10}],"x":2,"y":9,"parents":[]},{"symbol":"?","children":[{"x":3,"y":10},{"x":4,"y":10}],"x":3,"y":9,"parents":[]},{"symbol":"M","children":[{"x":4,"y":10}],"x":4,"y":9,"parents":[]},{"symbol":"M","children":[{"x":2,"y":11}],"x":1,"y":10,"parents":[]},{"symbol":"R","children":[{"x":2,"y":11},{"x":3,"y":11}],"x":3,"y":10,"parents":[]},{"symbol":"M","children":[{"x":3,"y":11},{"x":5,"y":11}],"x":4,"y":10,"parents":[]},{"symbol":"M","children":[{"x":1,"y":12}],"x":2,"y":11,"parents":[]},{"symbol":"?","children":[{"x":2,"y":12},{"x":3,"y":12},{"x":4,"y":12}],"x":3,"y":11,"parents":[]},{"symbol":"R","children":[{"x":5,"y":12}],"x":5,"y":11,"parents":[]},{"symbol":"M","children":[{"x":0,"y":13},{"x":2,"y":13}],"x":1,"y":12,"parents":[]},{"symbol":"M","children":[{"x":2,"y":13}],"x":2,"y":12,"parents":[]},{"symbol":"R","children":[{"x":3,"y":13}],"x":3,"y":12,"parents":[]},{"symbol":"?","children":[{"x":3,"y":13}],"x":4,"y":12,"parents":[]},{"symbol":"M","children":[{"x":5,"y":13}],"x":5,"y":12,"parents":[]},{"symbol":"M","children":[{"x":0,"y":14}],"x":0,"y":13,"parents":[]},{"symbol":"E","children":[{"x":2,"y":14}],"x":2,"y":13,"parents":[]},{"symbol":"M","children":[{"x":3,"y":14},{"x":4,"y":14}],"x":3,"y":13,"parents":[]},{"symbol":"M","children":[{"x":5,"y":14}],"x":5,"y":13,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":0,"y":14,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":2,"y":14,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":3,"y":14,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":4,"y":14,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":5,"y":14,"parents":[]}],"room_type":"EmptyRoom"}}'
NO_ZERO_X = '{"available_commands":["choose","return","key","click","wait","state"],"ready_for_command":true,"in_game":true,"game_state":{"choice_list":["x\u003d1","x\u003d3","x\u003d5","x\u003d6"],"screen_type":"MAP","screen_state":{"first_node_chosen":false,"current_node":{"x":0,"y":-1},"boss_available":false,"next_nodes":[{"symbol":"M","x":1,"y":0},{"symbol":"M","x":3,"y":0},{"symbol":"M","x":5,"y":0},{"symbol":"M","x":6,"y":0}]},"seed":1937920921210283156,"deck":[{"exhausts":false,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"a310e061-018a-4a03-801b-c16b0ee6c527","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"d3daf0d9-050f-4a5a-9093-414e46e187e8","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"09bbd1fa-8811-4552-b311-702f231c73b8","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"503a3bde-955f-47ab-9886-6e1187a30366","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"cost":1,"name":"Strike","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"389deaad-443e-4c40-b9f3-02d750e0c29a","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"cost":1,"name":"Defend","id":"Defend_R","type":"SKILL","ethereal":false,"uuid":"5f5c4198-2b36-4a46-8011-745acab86ff9","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"cost":1,"name":"Defend","id":"Defend_R","type":"SKILL","ethereal":false,"uuid":"57592a18-e261-4559-a8c5-5288e10d0b8e","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"cost":1,"name":"Defend","id":"Defend_R","type":"SKILL","ethereal":false,"uuid":"ea650f94-332d-4655-9407-e9d1c6649b1b","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"cost":1,"name":"Defend","id":"Defend_R","type":"SKILL","ethereal":false,"uuid":"06e2a317-61e6-4637-939e-41305a25223a","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"cost":2,"name":"Bash","id":"Bash","type":"ATTACK","ethereal":false,"uuid":"ae7b86d3-2272-47b9-ad6f-09d3882dc549","upgrades":0,"rarity":"BASIC","has_target":true}],"relics":[{"name":"Burning Blood","id":"Burning Blood","counter":-1},{"name":"Neow\u0027s Lament","id":"NeowsBlessing","counter":3}],"max_hp":80,"act_boss":"Hexaghost","gold":99,"action_phase":"WAITING_ON_USER","act":1,"screen_name":"MAP","room_phase":"COMPLETE","is_screen_up":true,"potions":[{"requires_target":false,"can_use":false,"can_discard":false,"name":"Potion Slot","id":"Potion Slot"},{"requires_target":false,"can_use":false,"can_discard":false,"name":"Potion Slot","id":"Potion Slot"},{"requires_target":false,"can_use":false,"can_discard":false,"name":"Potion Slot","id":"Potion Slot"}],"current_hp":80,"floor":0,"ascension_level":0,"class":"IRONCLAD","map":[{"symbol":"M","children":[{"x":2,"y":1}],"x":1,"y":0,"parents":[]},{"symbol":"M","children":[{"x":4,"y":1}],"x":3,"y":0,"parents":[]},{"symbol":"M","children":[{"x":5,"y":1}],"x":5,"y":0,"parents":[]},{"symbol":"M","children":[{"x":6,"y":1}],"x":6,"y":0,"parents":[]},{"symbol":"M","children":[{"x":3,"y":2}],"x":2,"y":1,"parents":[]},{"symbol":"?","children":[{"x":3,"y":2}],"x":4,"y":1,"parents":[]},{"symbol":"?","children":[{"x":4,"y":2}],"x":5,"y":1,"parents":[]},{"symbol":"M","children":[{"x":6,"y":2}],"x":6,"y":1,"parents":[]},{"symbol":"M","children":[{"x":2,"y":3},{"x":3,"y":3}],"x":3,"y":2,"parents":[]},{"symbol":"M","children":[{"x":4,"y":3}],"x":4,"y":2,"parents":[]},{"symbol":"M","children":[{"x":5,"y":3}],"x":6,"y":2,"parents":[]},{"symbol":"M","children":[{"x":2,"y":4}],"x":2,"y":3,"parents":[]},{"symbol":"?","children":[{"x":3,"y":4},{"x":4,"y":4}],"x":3,"y":3,"parents":[]},{"symbol":"M","children":[{"x":4,"y":4}],"x":4,"y":3,"parents":[]},{"symbol":"M","children":[{"x":4,"y":4}],"x":5,"y":3,"parents":[]},{"symbol":"?","children":[{"x":3,"y":5}],"x":2,"y":4,"parents":[]},{"symbol":"?","children":[{"x":3,"y":5}],"x":3,"y":4,"parents":[]},{"symbol":"$","children":[{"x":3,"y":5},{"x":5,"y":5}],"x":4,"y":4,"parents":[]},{"symbol":"E","children":[{"x":2,"y":6},{"x":3,"y":6}],"x":3,"y":5,"parents":[]},{"symbol":"R","children":[{"x":4,"y":6},{"x":5,"y":6}],"x":5,"y":5,"parents":[]},{"symbol":"?","children":[{"x":1,"y":7},{"x":3,"y":7}],"x":2,"y":6,"parents":[]},{"symbol":"M","children":[{"x":3,"y":7}],"x":3,"y":6,"parents":[]},{"symbol":"E","children":[{"x":4,"y":7}],"x":4,"y":6,"parents":[]},{"symbol":"?","children":[{"x":5,"y":7}],"x":5,"y":6,"parents":[]},{"symbol":"?","children":[{"x":2,"y":8}],"x":1,"y":7,"parents":[]},{"symbol":"$","children":[{"x":3,"y":8}],"x":3,"y":7,"parents":[]},{"symbol":"?","children":[{"x":3,"y":8}],"x":4,"y":7,"parents":[]},{"symbol":"M","children":[{"x":4,"y":8}],"x":5,"y":7,"parents":[]},{"symbol":"T","children":[{"x":3,"y":9}],"x":2,"y":8,"parents":[]},{"symbol":"T","children":[{"x":3,"y":9},{"x":4,"y":9}],"x":3,"y":8,"parents":[]},{"symbol":"T","children":[{"x":4,"y":9}],"x":4,"y":8,"parents":[]},{"symbol":"M","children":[{"x":3,"y":10}],"x":3,"y":9,"parents":[]},{"symbol":"E","children":[{"x":3,"y":10},{"x":4,"y":10}],"x":4,"y":9,"parents":[]},{"symbol":"M","children":[{"x":3,"y":11},{"x":4,"y":11}],"x":3,"y":10,"parents":[]},{"symbol":"R","children":[{"x":5,"y":11}],"x":4,"y":10,"parents":[]},{"symbol":"M","children":[{"x":3,"y":12}],"x":3,"y":11,"parents":[]},{"symbol":"E","children":[{"x":4,"y":12}],"x":4,"y":11,"parents":[]},{"symbol":"M","children":[{"x":4,"y":12},{"x":6,"y":12}],"x":5,"y":11,"parents":[]},{"symbol":"M","children":[{"x":2,"y":13},{"x":3,"y":13},{"x":4,"y":13}],"x":3,"y":12,"parents":[]},{"symbol":"?","children":[{"x":4,"y":13},{"x":5,"y":13}],"x":4,"y":12,"parents":[]},{"symbol":"M","children":[{"x":6,"y":13}],"x":6,"y":12,"parents":[]},{"symbol":"M","children":[{"x":3,"y":14}],"x":2,"y":13,"parents":[]},{"symbol":"M","children":[{"x":3,"y":14}],"x":3,"y":13,"parents":[]},{"symbol":"M","children":[{"x":4,"y":14}],"x":4,"y":13,"parents":[]},{"symbol":"M","children":[{"x":5,"y":14}],"x":5,"y":13,"parents":[]},{"symbol":"M","children":[{"x":5,"y":14}],"x":6,"y":13,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":3,"y":14,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":4,"y":14,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":5,"y":14,"parents":[]}],"room_type":"NeowRoom"}}'


class PathHandlerTestCase(unittest.TestCase):

    def test_can_handler(self):
        handler = PathHandler()
        state = GameState(json.loads(CHOOSE_ELITES))
        self.assertTrue(handler.can_handle(state))

    def test_no_errors_on_basic(self):
        handler = PathHandler()
        state = GameState(json.loads(STATE))

        self.assertEqual(["choose 0"], handler.handle(state))

    def test_choose_different_path_based_on_elites(self):
        handler = PathHandler()
        state = GameState(json.loads(CHOOSE_ELITES))

        self.assertEqual(["choose 1"], handler.handle(state))

    def test_initial_state(self):
        handler = PathHandler()
        state = GameState(json.loads(INITIAL_STATE))

        self.assertEqual(["choose 0"], handler.handle(state))

    def test_act2_start(self):
        handler = PathHandler()
        state = GameState(json.loads(ACT_2_START))

        self.assertEqual(["choose 2"], handler.handle(state))

    def test_no_zero_x(self):
        handler = PathHandler()
        state = GameState(json.loads(NO_ZERO_X))
        self.assertEqual(["choose 0"], handler.handle(state))


if __name__ == '__main__':
    unittest.main()
