from calculator.calculator_test_fixture import CalculatorTestFixture
from rs.calculator.cards import CardId, get_card
from rs.calculator.powers import PowerId
from rs.calculator.relics import RelicId


class CalculatorCardsTest(CalculatorTestFixture):

    def test_correct_statuses_lose_stacks_after_turn_end(self):
        pass

    def test_strength_adds_to_damage(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.STRENGTH: 4})
        play = self.when_calculating_state_play(state)
        self.see_enemy_lost_hp(play, 10)

    def test_strength_adds_to_multi_attack(self):
        state = self.given_state(CardId.TWIN_STRIKE, player_powers={PowerId.STRENGTH: 3})
        play = self.when_calculating_state_play(state)
        self.see_enemy_lost_hp(play, 16)

    def test_strength_when_negative(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.STRENGTH: -1})
        play = self.when_calculating_state_play(state)
        self.see_enemy_lost_hp(play, 5)

    def test_strength_when_damage_below_zero(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.STRENGTH: -100})
        play = self.when_calculating_state_play(state)
        self.see_enemy_lost_hp(play, 0)
        self.see_cards_played(play, 1)

    def test_strength_does_not_add_to_self_damage(self):
        state = self.given_state(CardId.BLOODLETTING, player_powers={PowerId.STRENGTH: 3})
        play = self.when_calculating_state_play(state)
        self.see_player_lost_hp(play, 3)

    def test_dexterity_adds_to_block(self):
        state = self.given_state(CardId.DEFEND_R, player_powers={PowerId.DEXTERITY: 3})
        play = self.when_calculating_state_play(state)
        self.see_player_has_block(play, 8)

    def test_dexterity_when_negative(self):
        state = self.given_state(CardId.DEFEND_R, player_powers={PowerId.DEXTERITY: -3})
        play = self.when_calculating_state_play(state)
        self.see_player_has_block(play, 2)

    def test_dexterity_when_block_would_be_below_zero(self):
        state = self.given_state(CardId.DEFEND_R, player_powers={PowerId.DEXTERITY: -13})
        play = self.when_calculating_state_play(state)
        self.see_player_has_block(play, 0)
        self.see_cards_played(play, 1)

    def test_vulnerable_when_attacking(self):
        state = self.given_state(CardId.STRIKE_R)
        state.monsters[0].powers[PowerId.VULNERABLE] = 1
        play = self.when_calculating_state_play(state)
        self.see_enemy_lost_hp(play, 9)

    def test_vulnerable_with_multi_attack_when_attacking(self):
        state = self.given_state(CardId.TWIN_STRIKE)
        state.monsters[0].powers[PowerId.VULNERABLE] = 1
        play = self.when_calculating_state_play(state)
        self.see_enemy_lost_hp(play, 14)

    def test_vulnerable_when_defending(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.VULNERABLE: 1})
        state.monsters[0].damage = 10
        state.monsters[0].hits = 1
        play = self.when_calculating_state_play(state)
        play.state.end_turn()
        self.see_player_lost_hp(play, 15)

    def test_vulnerable_with_multi_attack_when_defending(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.VULNERABLE: 1})
        state.monsters[0].damage = 7
        state.monsters[0].hits = 2
        play = self.when_calculating_state_play(state)
        play.state.end_turn()
        self.see_player_lost_hp(play, 20)

    def test_weak_when_attacking(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.WEAKENED: 1})
        play = self.when_calculating_state_play(state)
        self.see_enemy_lost_hp(play, 4)

    def test_weak_with_multi_attack_when_attacking(self):
        state = self.given_state(CardId.TWIN_STRIKE, player_powers={PowerId.WEAKENED: 1})
        play = self.when_calculating_state_play(state)
        self.see_enemy_lost_hp(play, 6)

    def test_weak_when_defending(self):
        pass

    def test_weak_with_multi_attack_when_defending(self):
        pass

    def test_frail(self):
        state = self.given_state(CardId.DEFEND_R, player_powers={PowerId.FRAIL: 1})
        play = self.when_calculating_state_play(state)
        self.see_player_has_block(play, 3)

    def test_entangled_no_attacks_played(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.ENTANGLED: 1})
        play = self.when_calculating_state_play(state)
        self.see_cards_played(play, 0)

    def test_vigor(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.VIGOR: 8})
        play = self.when_calculating_state_play(state)
        self.see_enemy_lost_hp(play, 14)
        self.see_player_does_not_have_power(play, PowerId.VIGOR)

    def test_curl_up(self):
        state = self.given_state(CardId.STRIKE_R)
        state.hand.append(get_card(CardId.STRIKE_R))
        state.monsters[0].powers[PowerId.CURL_UP] = 8
        play = self.when_calculating_state_play(state)
        self.see_cards_played(play, 2)
        self.see_enemy_lost_hp(play, 6)
        self.see_enemy_block_is(play, 2)

    def test_artifact_blocks_debuff(self):
        state = self.given_state(CardId.BASH)
        state.monsters[0].powers[PowerId.ARTIFACT] = 1
        play = self.when_calculating_state_play(state)
        self.see_enemy_lost_hp(play, 8)
        self.see_enemy_has_power(play, PowerId.VULNERABLE, 0)
        self.see_enemy_has_power(play, PowerId.ARTIFACT, 0)

    def test_artifact_blocks_negative_buff(self):
        state = self.given_state(CardId.DARK_SHACKLES)
        state.monsters[0].powers[PowerId.ARTIFACT] = 1
        play = self.when_calculating_state_play(state)
        self.see_enemy_does_not_have_power(play, PowerId.ARTIFACT)
        self.see_enemy_does_not_have_power(play, PowerId.STRENGTH)

    def test_artifact_does_not_block_buff(self):
        state = self.given_state(CardId.INFLAME, player_powers={PowerId.ARTIFACT: 1})
        play = self.when_calculating_state_play(state)
        self.see_player_has_power(play, PowerId.ARTIFACT, 1)
        self.see_player_has_power(play, PowerId.STRENGTH, 2)

    def test_artifact_multiple_debuffs(self):
        state = self.given_state(CardId.UPPERCUT)
        state.monsters[0].powers[PowerId.ARTIFACT] = 1
        play = self.when_calculating_state_play(state)
        self.see_enemy_lost_hp(play, 13)
        self.see_enemy_has_power(play, PowerId.VULNERABLE, 1)
        self.see_enemy_has_power(play, PowerId.WEAKENED, 0)
        self.see_enemy_has_power(play, PowerId.ARTIFACT, 0)

    def test_artifact_multiple_stacks(self):
        state = self.given_state(CardId.UPPERCUT)
        state.monsters[0].powers[PowerId.ARTIFACT] = 3
        play = self.when_calculating_state_play(state)
        self.see_enemy_lost_hp(play, 13)
        self.see_enemy_has_power(play, PowerId.VULNERABLE, 0)
        self.see_enemy_has_power(play, PowerId.WEAKENED, 0)
        self.see_enemy_has_power(play, PowerId.ARTIFACT, 1)

    def test_plated_armor_adds_block(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.PLATED_ARMOR: 4})
        play = self.when_calculating_state_play(state)
        play.end_turn()
        self.see_player_has_block(play, 4)

    def test_plated_armor_gets_reduced_by_damage(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.PLATED_ARMOR: 4})
        state.player.block = 0
        state.monsters[0].damage = 5
        state.monsters[0].hits = 2
        play = self.when_calculating_state_play(state)
        play.end_turn()
        self.see_player_lost_hp(play, 6)
        self.see_player_has_power(play, PowerId.PLATED_ARMOR, 2)

    def test_buffer_blocks_incoming_damage(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.BUFFER: 1})
        state.monsters[0].damage = 999
        state.monsters[0].hits = 1
        play = self.when_calculating_state_play(state)
        play.end_turn()
        self.see_player_lost_hp(play, 0)
        self.see_player_does_not_have_power(play, PowerId.BUFFER)

    def test_buffer_consumed_by_self_damage(self):
        state = self.given_state(CardId.BLOODLETTING, player_powers={PowerId.BUFFER: 1})
        state.monsters[0].damage = 8
        state.monsters[0].hits = 1
        play = self.when_calculating_state_play(state)
        play.end_turn()
        self.see_player_lost_hp(play, 8)
        self.see_player_does_not_have_power(play, PowerId.BUFFER)

    def test_multiple_buffer_stacks(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.BUFFER: 3})
        state.monsters[0].damage = 1
        state.monsters[0].hits = 10
        play = self.when_calculating_state_play(state)
        play.end_turn()
        self.see_player_lost_hp(play, 7)
        self.see_player_does_not_have_power(play, PowerId.BUFFER)

    def test_rage_adds_block_for_attack(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.RAGE: 3})
        state.hand.append(get_card(CardId.STRIKE_R))
        play = self.when_calculating_state_play(state)
        self.see_player_has_block(play, 6)

    def test_rage_does_not_add_block_for_skill(self):
        state = self.given_state(CardId.BLOODLETTING, player_powers={PowerId.RAGE: 3})
        play = self.when_calculating_state_play(state)
        self.see_player_has_block(play, 0)

    def test_metallicize_adds_block(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.METALLICIZE: 3})
        play = self.when_calculating_state_play(state)
        play.state.end_turn()
        self.see_player_has_block(play, 3)

    def test_metallicize_adds_block_stacking_with_orichalcum(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.METALLICIZE: 3},
                                 relics={RelicId.ORICHALCUM: 1})
        play = self.when_calculating_state_play(state)
        play.state.end_turn()
        self.see_player_has_block(play, 9)

    def test_intangible_blocks_all_but_one_damage(self):
        state = self.given_state(CardId.STRIKE_R, targets=2, player_powers={PowerId.INTANGIBLE: 1})
        state.monsters[0].damage = 999
        state.monsters[0].hits = 1
        state.monsters[1].damage = 1
        state.monsters[1].hits = 5
        play = self.when_calculating_state_play(state)
        play.state.end_turn()
        self.see_player_lost_hp(play, 6)

    def test_intangible_with_tungsten_rod_blocks_all_damage(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.INTANGIBLE: 1},
                                 relics={RelicId.TUNGSTEN_ROD: 1})
        state.monsters[0].damage = 999
        state.monsters[0].hits = 20
        play = self.when_calculating_state_play(state)
        play.state.end_turn()
        self.see_player_lost_hp(play, 0)

    def test_intangible_blocks_self_damage(self):
        state = self.given_state(CardId.BLOODLETTING, targets=2, player_powers={PowerId.INTANGIBLE: 1})
        play = self.when_calculating_state_play(state)
        self.see_player_lost_hp(play, 1)

    def test_flame_barrier_deals_damage_to_attacker(self):
        state = self.given_state(CardId.FLAME_BARRIER)
        state.monsters[0].damage = 1
        state.monsters[0].hits = 4
        play = self.when_calculating_state_play(state)
        play.end_turn()
        self.see_enemy_lost_hp(play, 16)

    def test_flame_barrier_blocked_by_block(self):
        state = self.given_state(CardId.FLAME_BARRIER)
        state.monsters[0].damage = 1
        state.monsters[0].hits = 4
        state.monsters[0].block = 10
        play = self.when_calculating_state_play(state)
        play.end_turn()
        self.see_enemy_lost_hp(play, 6)

    def test_attacker_dies_to_flame_barrier_and_then_their_attack_stops(self):
        state = self.given_state(CardId.FLAME_BARRIER)
        state.monsters[0].damage = 1
        state.monsters[0].hits = 999
        state.monsters[0].current_hp = 20
        play = self.when_calculating_state_play(state)
        play.end_turn()
        self.see_player_lost_hp(play, 0)
        self.see_enemy_hp_is(play, 0)

    def test_thorns_deals_damage(self):
        state = self.given_state(CardId.DEFEND_R, player_powers={PowerId.THORNS: 3})
        state.monsters[0].damage = 1
        state.monsters[0].hits = 4
        play = self.when_calculating_state_play(state)
        play.end_turn()
        self.see_enemy_lost_hp(play, 12)

    def test_sharp_hide_deals_damage_only_on_attack_play(self):
        state = self.given_state(CardId.TWIN_STRIKE)
        state.monsters[0].powers[PowerId.SHARP_HIDE] = 4
        play = self.when_calculating_state_play(state)
        self.see_player_lost_hp(play, 4)

    def test_attacking_angry_gives_strength(self):
        state = self.given_state(CardId.TWIN_STRIKE)
        state.monsters[0].powers[PowerId.ANGRY] = 3
        state.monsters[0].damage = 1
        state.monsters[0].hits = 2
        play = self.when_calculating_state_play(state)
        play.end_turn()
        self.see_player_lost_hp(play, 14)

    def test_flight_reduces_damage(self):
        state = self.given_state(CardId.TWIN_STRIKE)
        state.monsters[0].powers[PowerId.FLIGHT] = 3
        play = self.when_calculating_state_play(state)
        self.see_enemy_lost_hp(play, 4)
        self.see_enemy_has_power(play, PowerId.FLIGHT, 1)

    def test_flight_popped_causes_stun(self):
        state = self.given_state(CardId.TWIN_STRIKE)
        state.monsters[0].powers[PowerId.FLIGHT] = 2
        state.monsters[0].damage = 999
        state.monsters[0].hits = 999
        play = self.when_calculating_state_play(state)
        play.end_turn()
        self.see_player_lost_hp(play, 0)
        self.see_enemy_lost_hp(play, 4)
        self.see_enemy_does_not_have_power(play, PowerId.FLIGHT)

    def test_no_draw_prevents_draw(self):
        state = self.given_state(CardId.POMMEL_STRIKE, player_powers={PowerId.NO_DRAW: 1})
        play = self.when_calculating_state_play(state)
        self.see_player_hand_count(play, 0)
        self.see_player_discard_count(play, 1)

    def test_no_draw_blocked_by_artifact(self):
        state = self.given_state(CardId.BATTLE_TRANCE, player_powers={PowerId.ARTIFACT: 1})
        play = self.when_calculating_state_play(state)
        self.see_player_hand_count(play, 3)
        self.see_player_discard_count(play, 1)
        self.see_player_does_not_have_power(play, PowerId.ARTIFACT)
        self.see_player_does_not_have_power(play, PowerId.NO_DRAW)

    def test_split_removes_enemy_attack(self):
        state = self.given_state(CardId.STRIKE_R)
        state.monsters[0].damage = 13
        state.monsters[0].hits = 2
        state.monsters[0].current_hp = 46
        state.monsters[0].max_hp = 80
        state.monsters[0].powers = {PowerId.SPLIT: 1}
        play = self.when_calculating_state_play(state)
        play.end_turn()
        self.see_enemy_does_not_have_power(play, PowerId.SPLIT)
        self.see_player_lost_hp(play, 0)

    def test_not_quite_split_does_nothing(self):
        state = self.given_state(CardId.STRIKE_R)
        state.monsters[0].damage = 13
        state.monsters[0].hits = 2
        state.monsters[0].current_hp = 47
        state.monsters[0].max_hp = 80
        state.monsters[0].powers = {PowerId.SPLIT: 1}
        play = self.when_calculating_state_play(state)
        play.end_turn()
        self.see_enemy_has_power(play, PowerId.SPLIT, 1)
        self.see_player_lost_hp(play, 26)

    def test_mode_shift_removes_enemy_attack(self):
        state = self.given_state(CardId.STRIKE_R)
        state.monsters[0].damage = 13
        state.monsters[0].hits = 2
        state.monsters[0].powers = {PowerId.MODE_SHIFT: 1}
        play = self.when_calculating_state_play(state)
        play.end_turn()
        self.see_enemy_does_not_have_power(play, PowerId.MODE_SHIFT)
        self.see_player_lost_hp(play, 0)
        self.see_enemy_block_is(play, 20)

    def test_mode_does_nothing_when_not_broken(self):
        state = self.given_state(CardId.STRIKE_R)
        state.monsters[0].damage = 13
        state.monsters[0].hits = 2
        state.monsters[0].powers = {PowerId.MODE_SHIFT: 8}
        play = self.when_calculating_state_play(state)
        play.end_turn()
        self.see_enemy_has_power(play, PowerId.MODE_SHIFT, 2)
        self.see_player_lost_hp(play, 26)
        self.see_enemy_block_is(play, 0)
        self.see_enemy_lost_hp(play, 6)

    def test_mode_shift_blocks(self):
        state = self.given_state(CardId.TWIN_STRIKE)
        state.monsters[0].damage = 13
        state.monsters[0].hits = 2
        state.monsters[0].powers = {PowerId.MODE_SHIFT: 5}
        play = self.when_calculating_state_play(state)
        play.end_turn()
        self.see_enemy_does_not_have_power(play, PowerId.MODE_SHIFT)
        self.see_enemy_lost_hp(play, 5)
        self.see_enemy_block_is(play, 15)