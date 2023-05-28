from calculator.calculator_test_fixture import CalculatorTestFixture
from rs.calculator.cards import get_card
from rs.calculator.enums.card_id import CardId
from rs.calculator.enums.power_id import PowerId
from rs.calculator.enums.relic_id import RelicId


class CalculatorCardsTest(CalculatorTestFixture):

    def test_correct_statuses_lose_stacks_after_turn_end(self):
        pass

    def test_strength_adds_to_damage(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.STRENGTH: 4})
        play = self.when_playing_the_first_card(state)
        self.see_enemy_lost_hp(play, 10)

    def test_strength_adds_to_multi_attack(self):
        state = self.given_state(CardId.TWIN_STRIKE, player_powers={PowerId.STRENGTH: 3})
        play = self.when_playing_the_first_card(state)
        self.see_enemy_lost_hp(play, 16)

    def test_strength_when_negative(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.STRENGTH: -1})
        play = self.when_playing_the_first_card(state)
        self.see_enemy_lost_hp(play, 5)

    def test_strength_when_damage_below_zero(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.STRENGTH: -100})
        play = self.when_playing_the_first_card(state)
        self.see_enemy_lost_hp(play, 0)
        self.see_cards_played(play, 1)

    def test_strength_does_not_add_to_self_damage(self):
        state = self.given_state(CardId.BLOODLETTING, player_powers={PowerId.STRENGTH: 3})
        play = self.when_playing_the_first_card(state)
        self.see_player_lost_hp(play, 3)

    def test_dexterity_adds_to_block(self):
        state = self.given_state(CardId.DEFEND_R, player_powers={PowerId.DEXTERITY: 3})
        play = self.when_playing_the_first_card(state)
        self.see_player_has_block(play, 8)

    def test_dexterity_when_negative(self):
        state = self.given_state(CardId.DEFEND_R, player_powers={PowerId.DEXTERITY: -3})
        play = self.when_playing_the_first_card(state)
        self.see_player_has_block(play, 2)

    def test_dexterity_when_block_would_be_below_zero(self):
        state = self.given_state(CardId.DEFEND_R, player_powers={PowerId.DEXTERITY: -13})
        play = self.when_playing_the_first_card(state)
        self.see_player_has_block(play, 0)
        self.see_cards_played(play, 1)

    def test_vulnerable_when_attacking(self):
        state = self.given_state(CardId.STRIKE_R)
        state.monsters[0].powers[PowerId.VULNERABLE] = 1
        play = self.when_playing_the_first_card(state)
        self.see_enemy_lost_hp(play, 9)

    def test_vulnerable_with_multi_attack_when_attacking(self):
        state = self.given_state(CardId.TWIN_STRIKE)
        state.monsters[0].powers[PowerId.VULNERABLE] = 1
        play = self.when_playing_the_first_card(state)
        self.see_enemy_lost_hp(play, 14)

    def test_vulnerable_when_defending(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.VULNERABLE: 1})
        state.monsters[0].damage = 10
        state.monsters[0].hits = 1
        play = self.when_playing_the_first_card(state)
        play.state.end_turn()
        self.see_player_lost_hp(play, 15)

    def test_vulnerable_with_multi_attack_when_defending(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.VULNERABLE: 1})
        state.monsters[0].damage = 7
        state.monsters[0].hits = 2
        play = self.when_playing_the_first_card(state)
        play.state.end_turn()
        self.see_player_lost_hp(play, 20)

    def test_weak_when_attacking(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.WEAKENED: 1})
        play = self.when_playing_the_first_card(state)
        self.see_enemy_lost_hp(play, 4)

    def test_weak_with_multi_attack_when_attacking(self):
        state = self.given_state(CardId.TWIN_STRIKE, player_powers={PowerId.WEAKENED: 1})
        play = self.when_playing_the_first_card(state)
        self.see_enemy_lost_hp(play, 6)

    def test_weak_when_defending(self):
        state = self.given_state(CardId.NEUTRALIZE)
        state.monsters[0].damage = 10
        state.monsters[0].hits = 1
        play = self.when_playing_the_first_card(state)
        self.see_enemy_lost_hp(play, 3)
        self.see_enemy_has_power(play, PowerId.WEAKENED, 1)
        play.end_turn()
        self.see_player_lost_hp(play, 7)

    def test_weak_when_defending_no_rounding_needed(self):
        state = self.given_state(CardId.NEUTRALIZE)
        state.monsters[0].damage = 20
        state.monsters[0].hits = 1
        play = self.when_playing_the_first_card(state)
        self.see_enemy_lost_hp(play, 3)
        self.see_enemy_has_power(play, PowerId.WEAKENED, 1)
        play.end_turn()
        self.see_player_lost_hp(play, 15)

    def test_weak_with_multi_attack_when_defending(self):
        state = self.given_state(CardId.NEUTRALIZE)
        state.monsters[0].damage = 20
        state.monsters[0].hits = 2
        play = self.when_playing_the_first_card(state)
        self.see_enemy_lost_hp(play, 3)
        self.see_enemy_has_power(play, PowerId.WEAKENED, 1)
        play.end_turn()
        self.see_player_lost_hp(play, 30)

    def test_frail(self):
        state = self.given_state(CardId.DEFEND_R, player_powers={PowerId.FRAIL: 1})
        play = self.when_playing_the_first_card(state)
        self.see_player_has_block(play, 3)

    def test_entangled_no_attacks_played(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.ENTANGLED: 1})
        play = self.when_playing_the_first_card(state)
        self.see_cards_played(play, 0)

    def test_vigor(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.VIGOR: 8})
        play = self.when_playing_the_first_card(state)
        self.see_enemy_lost_hp(play, 14)
        self.see_player_does_not_have_power(play, PowerId.VIGOR)

    def test_curl_up(self):
        state = self.given_state(CardId.STRIKE_R)
        state.hand.append(get_card(CardId.STRIKE_R))
        state.monsters[0].powers[PowerId.CURL_UP] = 8
        play = self.when_playing_the_whole_hand(state)
        self.see_cards_played(play, 2)
        self.see_enemy_lost_hp(play, 6)
        self.see_enemy_block_is(play, 2)

    def test_artifact_blocks_debuff(self):
        state = self.given_state(CardId.BASH)
        state.monsters[0].powers[PowerId.ARTIFACT] = 1
        play = self.when_playing_the_first_card(state)
        self.see_enemy_lost_hp(play, 8)
        self.see_enemy_has_power(play, PowerId.VULNERABLE, 0)
        self.see_enemy_has_power(play, PowerId.ARTIFACT, 0)

    def test_artifact_blocks_negative_buff(self):
        state = self.given_state(CardId.DARK_SHACKLES)
        state.monsters[0].powers[PowerId.ARTIFACT] = 1
        play = self.when_playing_the_first_card(state)
        self.see_enemy_does_not_have_power(play, PowerId.ARTIFACT)
        self.see_enemy_does_not_have_power(play, PowerId.STRENGTH)

    def test_artifact_does_not_block_buff(self):
        state = self.given_state(CardId.INFLAME, player_powers={PowerId.ARTIFACT: 1})
        play = self.when_playing_the_first_card(state)
        self.see_player_has_power(play, PowerId.ARTIFACT, 1)
        self.see_player_has_power(play, PowerId.STRENGTH, 2)

    def test_artifact_multiple_debuffs(self):
        state = self.given_state(CardId.UPPERCUT)
        state.monsters[0].powers[PowerId.ARTIFACT] = 1
        play = self.when_playing_the_first_card(state)
        self.see_enemy_lost_hp(play, 13)
        self.see_enemy_has_power(play, PowerId.VULNERABLE, 1)
        self.see_enemy_has_power(play, PowerId.WEAKENED, 0)
        self.see_enemy_has_power(play, PowerId.ARTIFACT, 0)

    def test_artifact_multiple_stacks(self):
        state = self.given_state(CardId.UPPERCUT)
        state.monsters[0].powers[PowerId.ARTIFACT] = 3
        play = self.when_playing_the_first_card(state)
        self.see_enemy_lost_hp(play, 13)
        self.see_enemy_has_power(play, PowerId.VULNERABLE, 0)
        self.see_enemy_has_power(play, PowerId.WEAKENED, 0)
        self.see_enemy_has_power(play, PowerId.ARTIFACT, 1)

    def test_plated_armor_adds_block(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.PLATED_ARMOR: 4})
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_player_has_block(play, 4)

    def test_plated_armor_gets_reduced_by_damage(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.PLATED_ARMOR: 4})
        state.player.block = 0
        state.monsters[0].damage = 5
        state.monsters[0].hits = 2
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_player_lost_hp(play, 6)
        self.see_player_has_power(play, PowerId.PLATED_ARMOR, 2)

    def test_buffer_blocks_incoming_damage(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.BUFFER: 1})
        state.monsters[0].damage = 999
        state.monsters[0].hits = 1
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_player_lost_hp(play, 0)
        self.see_player_does_not_have_power(play, PowerId.BUFFER)

    def test_buffer_consumed_by_self_damage(self):
        state = self.given_state(CardId.BLOODLETTING, player_powers={PowerId.BUFFER: 1})
        state.monsters[0].damage = 8
        state.monsters[0].hits = 1
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_player_lost_hp(play, 8)
        self.see_player_does_not_have_power(play, PowerId.BUFFER)

    def test_multiple_buffer_stacks(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.BUFFER: 3})
        state.monsters[0].damage = 1
        state.monsters[0].hits = 10
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_player_lost_hp(play, 7)
        self.see_player_does_not_have_power(play, PowerId.BUFFER)

    def test_rage_adds_block_for_attack(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.RAGE: 3})
        state.hand.append(get_card(CardId.STRIKE_R))
        play = self.when_playing_the_whole_hand(state)
        self.see_player_has_block(play, 6)

    def test_rage_does_not_add_block_for_skill(self):
        state = self.given_state(CardId.BLOODLETTING, player_powers={PowerId.RAGE: 3})
        play = self.when_playing_the_first_card(state)
        self.see_player_has_block(play, 0)

    def test_metallicize_adds_block(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.METALLICIZE: 3})
        play = self.when_playing_the_first_card(state)
        play.state.end_turn()
        self.see_player_has_block(play, 3)

    def test_metallicize_adds_block_stacking_with_orichalcum(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.METALLICIZE: 3},
                                 relics={RelicId.ORICHALCUM: 1})
        play = self.when_playing_the_first_card(state)
        play.state.end_turn()
        self.see_player_has_block(play, 9)

    def test_intangible_player_blocks_all_but_one_damage(self):
        state = self.given_state(CardId.STRIKE_R, targets=2, player_powers={PowerId.INTANGIBLE_PLAYER: 1})
        state.monsters[0].damage = 999
        state.monsters[0].hits = 1
        state.monsters[1].damage = 1
        state.monsters[1].hits = 5
        play = self.when_playing_the_first_card(state)
        play.state.end_turn()
        self.see_player_lost_hp(play, 6)

    def test_intangible_player_with_tungsten_rod_blocks_all_damage(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.INTANGIBLE_PLAYER: 1},
                                 relics={RelicId.TUNGSTEN_ROD: 1})
        state.monsters[0].damage = 999
        state.monsters[0].hits = 20
        play = self.when_playing_the_first_card(state)
        play.state.end_turn()
        self.see_player_lost_hp(play, 0)

    def test_intangible_player_blocks_self_damage(self):
        state = self.given_state(CardId.BLOODLETTING, targets=2, player_powers={PowerId.INTANGIBLE_PLAYER: 1})
        play = self.when_playing_the_first_card(state)
        self.see_player_lost_hp(play, 1)

    def test_intangible_enemy_blocks_all_but_one_damage(self):
        state = self.given_state(CardId.TWIN_STRIKE)
        state.monsters[0].powers[PowerId.INTANGIBLE_ENEMY] = 1
        play = self.when_playing_the_first_card(state)
        self.see_enemy_lost_hp(play, 2)

    def test_flame_barrier_deals_damage_to_attacker(self):
        state = self.given_state(CardId.FLAME_BARRIER)
        state.monsters[0].damage = 1
        state.monsters[0].hits = 4
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_enemy_lost_hp(play, 16)

    def test_flame_barrier_blocked_by_block(self):
        state = self.given_state(CardId.FLAME_BARRIER)
        state.monsters[0].damage = 1
        state.monsters[0].hits = 4
        state.monsters[0].block = 10
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_enemy_lost_hp(play, 6)

    def test_attacker_dies_to_flame_barrier_and_then_their_attack_stops(self):
        state = self.given_state(CardId.FLAME_BARRIER)
        state.monsters[0].damage = 1
        state.monsters[0].hits = 999
        state.monsters[0].current_hp = 20
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_player_lost_hp(play, 0)
        self.see_enemy_hp_is(play, 0)

    def test_thorns_deals_damage(self):
        state = self.given_state(CardId.DEFEND_R, player_powers={PowerId.THORNS: 3})
        state.monsters[0].damage = 1
        state.monsters[0].hits = 4
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_enemy_lost_hp(play, 12)

    def test_thorns_deals_damage_when_attacked_for_0(self):
        state = self.given_state(CardId.DEFEND_R, player_powers={PowerId.THORNS: 3})
        state.monsters[0].damage = 0
        state.monsters[0].hits = 2
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_enemy_lost_hp(play, 6)

    def test_thorns_will_not_prevent_damage(self):
        state = self.given_state(CardId.WOUND, player_powers={PowerId.THORNS: 3})
        state.monsters[0].current_hp = 3
        state.monsters[0].damage = 5
        state.monsters[0].hits = 1
        play = self.when_playing_the_whole_hand(state)
        play.end_turn()
        self.see_enemy_hp_is(play, 0)
        self.see_player_lost_hp(play, 5)

    def test_sharp_hide_deals_damage_only_on_attack_play(self):
        state = self.given_state(CardId.TWIN_STRIKE)
        state.monsters[0].powers[PowerId.SHARP_HIDE] = 4
        play = self.when_playing_the_first_card(state)
        self.see_player_lost_hp(play, 4)

    def test_attacking_angry_gives_strength(self):
        state = self.given_state(CardId.TWIN_STRIKE)
        state.monsters[0].powers[PowerId.ANGRY] = 3
        state.monsters[0].damage = 1
        state.monsters[0].hits = 2
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_player_lost_hp(play, 14)

    def test_playing_a_skill_when_anger_nob_present_gives_strength(self):
        state = self.given_state(CardId.DEFEND_G)
        state.monsters[0].powers[PowerId.ANGER_NOB] = 2
        state.monsters[0].damage = 5
        state.monsters[0].hits = 1
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_player_lost_hp(play, 2)

    def test_only_the_monster_with_anger_nob_gets_strength_up(self):
        state = self.given_state(CardId.DEFEND_G, targets=2)
        state.monsters[0].powers[PowerId.ANGER_NOB] = 2
        state.monsters[0].damage = 5
        state.monsters[0].hits = 1
        state.monsters[1].damage = 5
        state.monsters[1].hits = 1
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_player_lost_hp(play, 7)

    def test_flight_reduces_damage(self):
        state = self.given_state(CardId.TWIN_STRIKE)
        state.monsters[0].powers[PowerId.FLIGHT] = 3
        play = self.when_playing_the_first_card(state)
        self.see_enemy_lost_hp(play, 4)
        self.see_enemy_has_power(play, PowerId.FLIGHT, 1)

    def test_flight_popped_causes_stun(self):
        state = self.given_state(CardId.TWIN_STRIKE)
        state.monsters[0].powers[PowerId.FLIGHT] = 2
        state.monsters[0].damage = 999
        state.monsters[0].hits = 999
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_player_lost_hp(play, 0)
        self.see_enemy_lost_hp(play, 4)
        self.see_enemy_does_not_have_power(play, PowerId.FLIGHT)

    def test_no_draw_prevents_draw(self):
        state = self.given_state(CardId.POMMEL_STRIKE, player_powers={PowerId.NO_DRAW: 1})
        play = self.when_playing_the_first_card(state)
        self.see_player_hand_count(play, 0)
        self.see_player_discard_pile_count(play, 1)

    def test_no_draw_blocked_by_artifact(self):
        state = self.given_state(CardId.BATTLE_TRANCE, player_powers={PowerId.ARTIFACT: 1})
        play = self.when_playing_the_first_card(state)
        self.see_player_hand_count(play, 3)
        self.see_player_discard_pile_count(play, 1)
        self.see_player_does_not_have_power(play, PowerId.ARTIFACT)
        self.see_player_does_not_have_power(play, PowerId.NO_DRAW)

    def test_split_removes_enemy_attack(self):
        state = self.given_state(CardId.STRIKE_R)
        state.monsters[0].damage = 13
        state.monsters[0].hits = 2
        state.monsters[0].current_hp = 46
        state.monsters[0].max_hp = 80
        state.monsters[0].powers = {PowerId.SPLIT: 1}
        play = self.when_playing_the_first_card(state)
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
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_enemy_has_power(play, PowerId.SPLIT, 1)
        self.see_player_lost_hp(play, 26)

    def test_mode_shift_removes_enemy_attack(self):
        state = self.given_state(CardId.STRIKE_R)
        state.monsters[0].damage = 13
        state.monsters[0].hits = 2
        state.monsters[0].powers = {PowerId.MODE_SHIFT: 1}
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_enemy_does_not_have_power(play, PowerId.MODE_SHIFT)
        self.see_player_lost_hp(play, 0)
        self.see_enemy_block_is(play, 20)

    def test_mode_does_nothing_when_not_broken(self):
        state = self.given_state(CardId.STRIKE_R)
        state.monsters[0].damage = 13
        state.monsters[0].hits = 2
        state.monsters[0].powers = {PowerId.MODE_SHIFT: 8}
        play = self.when_playing_the_first_card(state)
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
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_enemy_does_not_have_power(play, PowerId.MODE_SHIFT)
        self.see_enemy_lost_hp(play, 5)
        self.see_enemy_block_is(play, 15)

    def test_a_thousand_cuts_damage(self):
        state = self.given_state(CardId.DEFEND_G, player_powers={PowerId.THOUSAND_CUTS: 1})
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_enemy_lost_hp(play, 1)

    def test_after_image_block(self):
        state = self.given_state(CardId.STRIKE_G, player_powers={PowerId.AFTER_IMAGE: 1})
        state.monsters[0].damage = 1
        state.monsters[0].hits = 1
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_player_lost_hp(play, 0)

    def test_time_warp_not_incremented_when_not_present(self):
        state = self.given_state(CardId.STRIKE_R)
        play = self.when_playing_the_first_card(state)
        self.see_enemy_does_not_have_power(play, PowerId.TIME_WARP)

    def test_time_warp_incremented_by_card_plays(self):
        state = self.given_state(CardId.STRIKE_R)
        state.hand.append(get_card(CardId.STRIKE_R))
        state.hand.append(get_card(CardId.STRIKE_R))
        state.monsters[0].powers[PowerId.TIME_WARP] = 0
        play = self.when_playing_the_whole_hand(state)
        self.see_enemy_lost_hp(play, 18)
        self.see_enemy_has_power(play, PowerId.TIME_WARP, 3)

    def test_time_warp_caps_card_plays(self):
        state = self.given_state(CardId.CLEAVE)
        state.hand.append(get_card(CardId.CLEAVE))
        state.hand.append(get_card(CardId.CLEAVE))
        state.monsters[0].powers[PowerId.TIME_WARP] = 10
        play = self.when_playing_the_whole_hand(state)
        # see that only 2 of the 3 cleaves are played because time warp stops it
        self.see_enemy_lost_hp(play, 16)
        self.see_enemy_has_power(play, PowerId.TIME_WARP, 12)

    def test_poison_damages_and_decrements(self):
        state = self.given_state(CardId.WOUND)
        state.monsters[0].powers[PowerId.POISON] = 3
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_enemy_lost_hp(play, 3)
        self.see_enemy_has_power(play, PowerId.POISON, 2)

    def test_poison_kills_and_so_prevents_damage(self):
        state = self.given_state(CardId.WOUND)
        state.monsters[0].powers[PowerId.POISON] = 3
        state.monsters[0].current_hp = 3
        state.monsters[0].damage = 7
        state.monsters[0].hits = 1
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_enemy_hp_is(play, 0)
        self.see_player_lost_hp(play, 0)

    def test_damaging_shifter_reduces_incoming_damage(self):
        state = self.given_state(CardId.STRIKE_R)
        state.monsters[0].powers[PowerId.SHIFTING] = 1
        state.monsters[0].damage = 7
        state.monsters[0].hits = 1
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_enemy_lost_hp(play, 6)
        self.see_player_lost_hp(play, 1)

    def test_damaging_shifter_more_complicated(self):
        state = self.given_state(CardId.NEUTRALIZE)
        state.monsters[0].powers[PowerId.SHIFTING] = 1
        state.monsters[0].powers[PowerId.STRENGTH] = -3
        state.monsters[0].damage = 20   # Strength-unadjusted damage
        state.monsters[0].hits = 1
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_enemy_lost_hp(play, 3)
        self.see_enemy_has_power(play, PowerId.STRENGTH, -6)
        self.see_player_lost_hp(play, 10)

    def test_damaging_shifter_reduces_incoming_damage_in_bigger_fight(self):
        state = self.given_state(CardId.STRIKE_R, targets=2)
        state.monsters[0].powers[PowerId.SHIFTING] = 1
        state.monsters[0].damage = 1
        state.monsters[0].hits = 1
        state.monsters[1].damage = 1
        state.monsters[1].hits = 1
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_enemy_lost_hp(play, 6, enemy_index=0)
        self.see_enemy_lost_hp(play, 0, enemy_index=1)
        self.see_player_lost_hp(play, 1)

    def test_damaging_non_shifter_does_not_reduces_incoming_damage_in_bigger_fight(self):
        state = self.given_state(CardId.STRIKE_R, targets=2)
        state.monsters[0].damage = 1
        state.monsters[0].hits = 1
        state.monsters[1].powers[PowerId.SHIFTING] = 1
        state.monsters[1].damage = 1
        state.monsters[1].hits = 1
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_enemy_lost_hp(play, 6, enemy_index=0)
        self.see_enemy_lost_hp(play, 0, enemy_index=1)
        self.see_player_lost_hp(play, 2)

    def test_damaging_all_in_bigger_fight_that_includes_shifter_does_reduce_damage(self):
        state = self.given_state(CardId.CLEAVE, targets=2)
        state.monsters[0].powers[PowerId.SHIFTING] = 1
        state.monsters[0].damage = 1
        state.monsters[0].hits = 1
        state.monsters[1].damage = 1
        state.monsters[1].hits = 1
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_enemy_lost_hp(play, 8, enemy_index=0)
        self.see_enemy_lost_hp(play, 8, enemy_index=1)
        self.see_player_lost_hp(play, 1)

    def test_constricted_does_damage(self):
        state = self.given_state(CardId.WOUND, player_powers={PowerId.CONSTRICTED: 10})
        state.monsters[0].damage = 1
        state.monsters[0].hits = 1
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_player_lost_hp(play, 11)

    def test_malleable_blocks(self):
        state = self.given_state(CardId.CLEAVE)
        state.monsters[0].powers[PowerId.MALLEABLE] = 3
        play = self.when_playing_the_first_card(state)
        self.see_enemy_has_power(play, PowerId.MALLEABLE, 4)
        self.see_enemy_lost_hp(play, 8)
        self.see_enemy_block_is(play, 4)

    def test_malleable_blocks_after_double_attack(self):
        state = self.given_state(CardId.TWIN_STRIKE)
        state.monsters[0].powers[PowerId.MALLEABLE] = 3
        play = self.when_playing_the_first_card(state)
        self.see_enemy_has_power(play, PowerId.MALLEABLE, 5)
        self.see_enemy_lost_hp(play, 10)
        self.see_enemy_block_is(play, 9)

    def test_malleable_blocks_after_triple_attack(self):
        state = self.given_state(CardId.EVISCERATE)
        state.monsters[0].powers[PowerId.MALLEABLE] = 3
        play = self.when_playing_the_first_card(state)
        self.see_enemy_has_power(play, PowerId.MALLEABLE, 6)
        self.see_enemy_lost_hp(play, 21)
        self.see_enemy_block_is(play, 15)

    def test_choked_causes_card_plays_to_damage(self):
        state = self.given_state(CardId.DEFEND_R)
        state.monsters[0].powers[PowerId.CHOKED] = 3
        play = self.when_playing_the_first_card(state)
        self.see_enemy_lost_hp(play, 3)

    def test_only_the_monster_with_choked_gets_damaged_by_it(self):
        state = self.given_state(CardId.DEFEND_G, targets=2)
        state.monsters[0].powers[PowerId.CHOKED] = 2
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_enemy_lost_hp(play, 2, enemy_index=0)
        self.see_enemy_lost_hp(play, 0, enemy_index=1)

    def test_envenom_power_does_not_work_on_blocked_hit(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.ENVENOM: 1})
        state.monsters[0].block = 10
        play = self.when_playing_the_first_card(state)
        self.see_player_has_power(play, PowerId.ENVENOM, 1)
        self.see_enemy_block_is(play, 4)
        self.see_enemy_lost_hp(play, 0)
        self.see_enemy_does_not_have_power(play, PowerId.POISON)

    def test_envenom_power_applies_poison_on_unblocked_hit(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.ENVENOM: 1})
        state.monsters[0].block = 5
        play = self.when_playing_the_first_card(state)
        self.see_player_has_power(play, PowerId.ENVENOM, 1)
        self.see_enemy_block_is(play, 0)
        self.see_enemy_lost_hp(play, 1)
        self.see_enemy_has_power(play, PowerId.POISON, 1)

    def test_envenom_power_applies_poison_on_unblocked_multi_hit(self):
        state = self.given_state(CardId.TWIN_STRIKE, player_powers={PowerId.ENVENOM: 1})
        play = self.when_playing_the_first_card(state)
        self.see_player_has_power(play, PowerId.ENVENOM, 1)
        self.see_enemy_lost_hp(play, 10)
        self.see_enemy_has_power(play, PowerId.POISON, 2)

    def test_envenom_power_stacks_onto_existing_poison(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.ENVENOM: 1})
        state.monsters[0].powers[PowerId.POISON] = 1
        play = self.when_playing_the_first_card(state)
        self.see_player_has_power(play, PowerId.ENVENOM, 1)
        self.see_enemy_lost_hp(play, 6)
        self.see_enemy_has_power(play, PowerId.POISON, 2)

    def test_corpse_explosion_deals_max_hp_damage_when_killed_actively(self):
        state = self.given_state(CardId.STRIKE_R, targets=2)
        state.monsters[0].powers[PowerId.CORPSE_EXPLOSION_POWER] = 1
        state.monsters[0].current_hp = 1
        state.monsters[0].max_hp = 10
        play = self.when_playing_the_first_card(state)
        self.see_enemy_hp_is(play, 0, enemy_index=0)
        self.see_enemy_lost_hp(play, state.monsters[0].max_hp, enemy_index=1)

    def test_corpse_explosion_deals_max_hp_damage_when_killed_passively(self):
        state = self.given_state(CardId.WOUND, targets=2)
        state.monsters[0].powers[PowerId.POISON] = 1
        state.monsters[0].powers[PowerId.CORPSE_EXPLOSION_POWER] = 1
        state.monsters[0].current_hp = 1
        state.monsters[0].max_hp = 50
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_enemy_hp_is(play, 0, enemy_index=0)
        self.see_enemy_lost_hp(play, state.monsters[0].max_hp, enemy_index=1)

    def test_corpse_explosion_stacked_deals_more_damage(self):
        state = self.given_state(CardId.WOUND, targets=2)
        state.monsters[0].powers[PowerId.POISON] = 1
        state.monsters[0].powers[PowerId.CORPSE_EXPLOSION_POWER] = 2
        state.monsters[0].current_hp = 1
        state.monsters[0].max_hp = 50
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_enemy_hp_is(play, 0, enemy_index=0)
        self.see_enemy_lost_hp(play, state.monsters[0].max_hp * state.monsters[0].powers[PowerId.CORPSE_EXPLOSION_POWER], enemy_index=1)

    def test_wraith_form_power_decreases_dexterity_on_turn_end(self):
        state = self.given_state(CardId.WOUND, player_powers={PowerId.WRAITH_FORM_POWER: 1})
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_player_has_power(play, PowerId.DEXTERITY, -1)

    def test_wraith_form_power_dexterity_losses_stack(self):
        state = self.given_state(CardId.WOUND, player_powers={PowerId.WRAITH_FORM_POWER: 1, PowerId.DEXTERITY: -3})
        play = self.when_playing_the_first_card(state)
        play.end_turn()
        self.see_player_has_power(play, PowerId.DEXTERITY, -4)

    def test_double_damage_doubles_damage(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.DOUBLE_DAMAGE: 1})
        play = self.when_playing_the_first_card(state)
        self.see_enemy_lost_hp(play, 12)

    def test_double_damage_doubles_damage_multi_hit(self):
        state = self.given_state(CardId.TWIN_STRIKE, player_powers={PowerId.DOUBLE_DAMAGE: 1})
        play = self.when_playing_the_first_card(state)
        self.see_enemy_lost_hp(play, 20)

    def test_double_damage_against_intangible(self):
        state = self.given_state(CardId.TWIN_STRIKE, player_powers={PowerId.DOUBLE_DAMAGE: 1})
        state.monsters[0].powers[PowerId.INTANGIBLE_ENEMY] = 1
        play = self.when_playing_the_first_card(state)
        self.see_enemy_lost_hp(play, 2)

    def test_juggernaut_block_from_card(self):
        state = self.given_state(CardId.DEFEND_R, player_powers={PowerId.JUGGERNAUT: 5})
        play = self.when_playing_the_first_card(state)
        self.see_enemy_lost_hp(play, 5)
        self.see_random_damage_dealt(play, 0)

    def test_juggernaut_block_from_power(self):
        state = self.given_state(CardId.WOUND, player_powers={PowerId.JUGGERNAUT: 7, PowerId.METALLICIZE: 4}, targets=2)
        play = self.when_playing_the_first_card(state)
        self.see_enemy_lost_hp(play, 0, enemy_index=0)
        self.see_enemy_lost_hp(play, 0, enemy_index=1)
        self.see_random_damage_dealt(play, 0)
        play.end_turn()
        self.see_enemy_lost_hp(play, 0, enemy_index=0)
        self.see_enemy_lost_hp(play, 0, enemy_index=1)
        self.see_random_damage_dealt(play, 7)

    def test_juggernaut_does_not_damage_when_0_block_gained(self):
        state = self.given_state(CardId.DEFEND_R, player_powers={PowerId.JUGGERNAUT: 5, PowerId.DEXTERITY: -5})
        play = self.when_playing_the_first_card(state)
        self.see_enemy_lost_hp(play, 0)
        self.see_random_damage_dealt(play, 0)

    def test_panache_decrements(self):
        state = self.given_state(CardId.DEFEND_R)
        state.player.powers[PowerId.PANACHE] = 5
        play = self.when_playing_the_first_card(state)
        self.see_enemy_lost_hp(play, 0)
        self.see_player_has_power(play, PowerId.PANACHE, 4)

    def test_panache_triggers(self):
        state = self.given_state(CardId.DEFEND_R)
        state.player.powers[PowerId.PANACHE] = 1
        play = self.when_playing_the_first_card(state)
        self.see_enemy_lost_hp(play, 10)
        self.see_player_has_power(play, PowerId.PANACHE, 5)

    def test_sadistic_triggers(self):
        state = self.given_state(CardId.BLIND, player_powers={PowerId.SADISTIC: 5})
        play = self.when_playing_the_first_card(state)
        self.see_enemy_has_power(play, PowerId.WEAKENED, 2)
        self.see_enemy_lost_hp(play, 5)

    def test_sadistic_triggers_on_multi_hit(self):
        state = self.given_state(CardId.BOUNCING_FLASK, player_powers={PowerId.SADISTIC: 5})
        play = self.when_playing_the_first_card(state)
        self.see_enemy_has_power(play, PowerId.POISON, 9)
        self.see_enemy_lost_hp(play, 15)

    def test_sadistic_envenom_interaction(self):
        state = self.given_state(CardId.STRIKE_R, player_powers={PowerId.SADISTIC: 5, PowerId.ENVENOM: 1})
        play = self.when_playing_the_first_card(state)
        self.see_enemy_has_power(play, PowerId.POISON, 1)
        self.see_enemy_lost_hp(play, 11)

    def test_sadistic_should_not_damage_player(self):
        state = self.given_state(CardId.DOUBT, player_powers={PowerId.SADISTIC: 5})
        play = self.when_playing_the_whole_hand(state)
        play.end_turn()
        self.see_player_has_power(play, PowerId.WEAKENED, 1)
        self.see_player_lost_hp(play, 0)

    def test_heatsinks_does_not_trigger_its_own_power_when_played(self):
        state = self.given_state(CardId.HEATSINKS)
        play = self.when_playing_the_first_card(state)
        self.see_player_has_power(play, PowerId.HEATSINK, 1)
        self.see_player_drew_cards(play, 0)

    def test_heatsink(self):
        state = self.given_state(CardId.INFLAME, player_powers={PowerId.HEATSINK: 2})
        play = self.when_playing_the_first_card(state)
        self.see_player_has_power(play, PowerId.STRENGTH, 2)
        self.see_player_drew_cards(play, 2)


