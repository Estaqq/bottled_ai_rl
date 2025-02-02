train = True

CARD_REMOVAL_PRIORITY_LIST = ['strike', 'strike+', 'defend', 'defend+']

DESIRED_CARDS_FOR_DECK: dict[str, int] = {
    'all for one': 2,
    'reprogram': 1,
    'claw': 5,
    'go for the eyes': 2,
    'beam cell': 2,
    'streamline': 1,
    'equilibrium': 1,
    'hyperbeam': 1,
    'reinforced body': 1,
    'ftl': 3,
    'scrape': 1,
    'dualcast': 1,
    'steam barrier': 3,
    'ball lightning': 1,
    'boot sequence': 1,
    'fission': 1,
    'zap': 1,
    'apparition': 1,
}

DESIRED_CARDS_FROM_POTIONS: dict[str, int] = {
    'echo form': 1,
    'apotheosis': 1,
    'all for one': 2,
    'machine learning': 3,
    'buffer': 2,
    'self repair': 2,
    'claw': 8,
    'core surge': 2,
    'defragment': 2,
    'loop': 1,
    'ftl': 5,

}

HIGH_PRIORITY_UPGRADES = [
    'Apotheosis',
]

DESIRED_POTIONS = [
    'fruit juice',
    'fairy in a bottle',
    'cultist potion',
    'power potion',
    'potion of capacity',
    'heart of iron',
    'duplication potion',
    'distilled chaos',
    'blessing of the forge',
    'attack potion',
    'dexterity potion',
    'ambrosia',
    'fear potion',
    'essence of steel',
    'strength potion',
    'regen potion',
    'blood potion',
    'entropic brew',
    'liquid bronze',
    'energy potion',
    'skill potion',
    'ancient potion',
    'weak potion',
    'gambler\u0027s brew',
    'poison potion',
    'colorless potion',
    'flex potion',
    'swift potion',
    'bottled miracle',
    'essence of darkness',
    'fire potion',
    'explosive potion',
    'focus potion',  # down-prioritized this since we intentionally lose a lot of Focus with this strategy
    'speed potion',
    'block potion',
    'cunning potion',
    'ghost in a jar',
    'stance potion',
    'smoke bomb',
    'elixir potion',
    'liquid memories',
    'snecko oil',
]