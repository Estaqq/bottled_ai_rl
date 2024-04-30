from enum import Enum


class CardId(Enum):
    # PLACEHOLDER/LOGIC CARDS
    FAKE = 'fake'  # temp fake card for all the ones we don't know yet in game. Basically, treat like a wound.
    DRAW_FREE_EARLY = 'draw free early'
    DRAW_FREE = 'draw free'
    DRAW_PAY_EARLY = 'draw pay early'
    DRAW_PAY = 'draw pay'

    # REAL CARDS
    A_THOUSAND_CUTS = 'a thousand cuts'
    ACCURACY = 'accuracy'
    ACROBATICS = 'acrobatics'
    ADRENALINE = 'adrenaline'
    AFTER_IMAGE = 'after image'
    AGGREGATE = 'aggregate'
    ALL_FOR_ONE = 'all for one'
    AMPLIFY = 'amplify'
    ANGER = 'anger'
    APOTHEOSIS = 'apotheosis'
    APPARITION = 'ghostly'
    ASCENDERS_BANE = 'ascender\u0027s bane'
    AUTO_SHIELDS = 'auto shields'
    BACKFLIP = 'backflip'
    BACKSTAB = 'backstab'
    BANDAGE_UP = 'bandage up'
    BALL_LIGHTNING = 'ball lightning'
    BANE = 'bane'
    BASH = 'bash'
    BARRAGE = 'barrage'
    BATTLE_HYMN = 'battle hymn'
    BATTLE_TRANCE = 'battle trance'
    BEAM_CELL = 'beam cell'
    BERSERK = 'berserk'
    BIASED_COGNITION = 'biased cognition'
    BITE = 'bite'
    BLADE_DANCE = 'blade dance'
    BLIND = 'blind'
    BLIZZARD = 'blizzard'
    BLOODLETTING = 'bloodletting'
    BLOOD_FOR_BLOOD = 'blood for blood'
    BLUDGEON = 'bludgeon'
    BLUR = 'blur'
    BODY_SLAM = 'body slam'
    BOOT_SEQUENCE = 'bootsequence'  # Weird id alert!
    BOUNCING_FLASK = 'bouncing flask'
    BOWLING_BASH = 'bowling bash'
    BUFFER = 'buffer'
    BULLET_TIME = 'bullet time'
    BULLSEYE = 'lockon'
    BURN = 'burn'
    BURST = 'burst'
    CALCULATED_GAMBLE = 'calculated gamble'
    CALTROPS = 'caltrops'
    CAPACITOR = 'capacitor'
    CARNAGE = 'carnage'
    CATALYST = 'catalyst'
    CHAOS = 'chaos'
    CHARGE_BATTERY = 'conserve battery'
    CHILL = 'chill'
    CHOKE = 'choke'
    CLASH = 'clash'
    CLAW = 'gash'  # Weird id alert!
    CLEAVE = 'cleave'
    CLOAK_AND_DAGGER = 'cloak and dagger'
    CLOTHESLINE = 'clothesline'
    CLUMSY = 'clumsy'
    COOLHEADED = 'coolheaded'
    COLD_SNAP = 'cold snap'
    COMPILE_DRIVER = 'compile driver'
    CONCENTRATE = 'concentrate'
    CONCLUDE = 'conclude'
    CONSECRATE = 'consecrate'
    CONSUME = 'consume'
    CORE_SURGE = 'core surge'
    CORPSE_EXPLOSION = 'corpse explosion'
    CORRUPTION = 'corruption'
    CREATIVE_AI = 'creative ai'
    CRESCENDO = 'crescendo'
    CRUSH_JOINTS = 'crush joints'
    CURSE_OF_THE_BELL = 'curseofthebell'  # Weird id alert!
    CRIPPLING_CLOUD = 'crippling poison'
    DAGGER_THROW = 'dagger throw'
    DAGGER_SPRAY = 'dagger spray'
    DARK_EMBRACE = 'dark embrace'
    DARK_SHACKLES = 'dark shackles'
    DARKNESS = 'darkness'
    DASH = 'dash'
    DAZED = 'dazed'
    DEADLY_POISON = 'deadly poison'
    DECAY = 'decay'
    DEEP_BREATH = 'deep breath'
    DEFEND_B = 'defend_b'
    DEFEND_G = 'defend_g'
    DEFEND_P = 'defend_p'
    DEFEND_R = 'defend_r'
    DEFLECT = 'deflect'
    DEFRAGMENT = 'defragment'
    DEMON_FORM = 'demon form'
    DEVOTION = 'devotion'
    DIE_DIE_DIE = 'die die die'
    DISARM = 'disarm'
    DRAMATIC_ENTRANCE = 'dramatic entrance'
    DROPKICK = 'dropkick'
    DOOM_AND_GLOOM = 'doom and gloom'
    DODGE_AND_ROLL = 'dodge and roll'
    DOUBLE_ENERGY = 'double energy'
    DOUBLE_TAP = 'double tap'
    DOUBT = 'doubt'
    DUALCAST = 'dualcast'
    ECHO_FORM = 'echo form'
    ELECTRODYNAMICS = 'electrodynamics'
    EMPTY_BODY = 'empty body'
    EMPTY_FIST = 'empty fist'
    EMPTY_MIND = 'empty mind'
    ENDLESS_AGONY = 'endless agony'  # Note: the special bits of this card aren't relevant for our current calculator.
    ENLIGHTENMENT = 'enlightenment'
    ENTRENCH = 'entrench'
    ENVENOM = 'envenom'
    EQUILIBRIUM = 'undo'  # Weird id alert!
    ERUPTION = 'eruption'
    ESCAPE_PLAN = 'escape plan'
    ESTABLISHMENT = 'establishment'
    EVISCERATE = 'eviscerate'
    EVOLVE = 'evolve'
    EXPERTISE = 'expertise'
    FEED = 'feed'
    FEEL_NO_PAIN = 'feel no pain'
    FIEND_FIRE = 'fiend fire'
    FINESSE = 'finesse'
    FINISHER = 'finisher'
    FIRE_BREATHING = 'fire breathing'
    FISSION = 'fission'
    FLAME_BARRIER = 'flame barrier'
    FLASH_OF_STEEL = 'flash of steel'
    FLECHETTES = 'flechettes'
    FLEX = 'flex'
    FLURRY_OF_BLOWS = 'flurry of blows'
    FLYING_KNEE = 'flying knee'
    FLYING_SLEEVES = 'flying sleeves'
    FOLLOW_UP = 'follow-up'
    FOOTWORK = 'footwork'
    FTL = 'ftl'
    FUSION = 'fusion'
    GHOSTLY_ARMOR = 'ghostly armor'
    GLACIER = 'glacier'
    GLASS_KNIFE = 'glass knife'
    GENETIC_ALGORITHM = 'genetic algorithm'
    GO_FOR_THE_EYES = 'go for the eyes'
    GOOD_INSTINCTS = 'good instincts'
    GRAND_FINALE = 'grand finale'
    HAND_OF_GREED = 'handofgreed'  # Weird id alert!
    HEATSINKS = 'heatsinks'
    HEAVY_BLADE = 'heavy blade'
    HEEL_HOOK = 'heel hook'
    HEMOKINESIS = 'hemokinesis'
    HELLO_WORLD = 'hello world'
    HYPERBEAM = 'hyperbeam'
    IMMOLATE = 'immolate'
    IMPATIENCE = 'impatience'
    IMPERVIOUS = 'impervious'
    INFINITE_BLADES = 'infinite blades'
    INFLAME = 'inflame'
    INJURY = 'injury'
    INSIGHT = 'insight'
    INTIMIDATE = 'intimidate'
    IRON_WAVE = 'iron wave'
    JAX = 'j.a.x.'
    JUDGEMENT = 'judgement'
    JUGGERNAUT = 'juggernaut'
    LEAP = 'leap'
    LEG_SWEEP = 'leg sweep'
    LIKE_WATER = 'like water'
    LIMIT_BREAK = 'limit break'
    LOOP = 'loop'
    MACHINE_LEARNING = 'machine learning'
    MAGNETISM = 'magnetism'
    MALAISE = 'malaise'
    MASTER_OF_STRATEGY = 'master of strategy'
    MAYHEM = 'mayhem'
    MELTER = 'melter'
    MENTAL_FORTRESS = 'mental fortress'
    METALLICIZE = 'metallicize'
    METEOR_STRIKE = 'meteor strike'
    MIND_BLAST = 'mind blast'
    MIRACLE = 'miracle'
    MULTI_CAST = 'multicast'
    NEUTRALIZE = 'neutralize'
    NOXIOUS_FUMES = 'noxious fumes'
    OFFERING = 'offering'
    OUTMANEUVER = 'outmaneuver'
    OVERCLOCK = 'steam power'
    PAIN = 'pain'
    PANACEA = 'panacea'
    PANACHE = 'panache'  # We currently have damage provided by triggering the Panache power hardcoded to 10. It's the first power we've run into that has multiple values associated with it.
    PARASITE = 'parasite'
    PERFECTED_STRIKE = 'perfected strike'
    PHANTASMAL_KILLER = 'phantasmal killer'
    PIERCING_WAIL = 'piercing wail'
    POISONED_STAB = 'poisoned stab'
    POMMEL_STRIKE = 'pommel strike'
    POWER_THROUGH = 'power through'
    PRAY = 'pray'
    PREDATOR = 'predator'
    PREPARED = 'prepared'
    PROSTRATE = 'prostrate'
    PROTECT = 'protect'
    PUMMEL = 'pummel'
    QUICK_SLASH = 'quick slash'
    RAGE = 'rage'
    RAGNAROK = 'ragnarok'
    RAINBOW = 'rainbow'
    RAMPAGE = 'rampage'
    REAPER = 'reaper'
    REBOOT = 'reboot'
    RECURSION = 'redo'
    REFLEX = 'reflex'
    REGRET = 'regret'
    RECKLESS_CHARGE = 'reckless charge'
    REINFORCED_BODY = 'reinforced body'
    REPROGRAM = 'reprogram'
    RIDDLE_WITH_HOLES = 'riddle with holes'
    RIP_AND_TEAR = 'rip and tear'
    RITUAL_DAGGER = 'ritual dagger'
    SADISTIC_NATURE = 'sadistic nature'
    SANCTITY = 'sanctity'
    SASH_WHIP = 'sash whip'
    SCRAPE = 'scrape'  # Ignores the fact that we throw away non-0 cards after draw
    SCRAWL = 'scrawl'
    SECOND_WIND = 'second wind'
    SEEING_RED = 'seeing red'
    SELF_REPAIR = 'self repair'
    SENTINEL = 'sentinel'
    SEVER_SOUL = 'sever soul'
    SHAME = 'shame'
    SHIV = 'shiv'
    SHOCKWAVE = 'shockwave'
    SHRUG_IT_OFF = 'shrug it off'
    SKEWER = 'skewer'
    SKIM = 'skim'
    SLICE = 'slice'
    SLIMED = 'slimed'
    SMITE = 'smite'
    SNEAKY_STRIKE = 'underhanded strike'  # Weird id alert!
    SPOT_WEAKNESS = 'spot weakness'
    STACK = 'stack'
    STEAM_BARRIER = 'steam'  # Weird id alert!
    STORM = 'storm'
    STORM_OF_STEEL = 'storm of steel'
    STREAMLINE = 'streamline'
    STRIKE_B = 'strike_b'
    STRIKE_G = 'strike_g'
    STRIKE_P = 'strike_p'
    STRIKE_R = 'strike_r'
    STUDY = 'study'
    SUCKER_PUNCH = 'sucker punch'
    SUNDER = 'sunder'
    SURVIVOR = 'survivor'
    SWEEPING_BEAM = 'sweeping beam'
    SWIFT_STRIKE = 'swift strike'
    SWORD_BOOMERANG = 'sword boomerang'
    TACTICIAN = 'tactician'
    TEMPEST = 'tempest'
    TERROR = 'terror'
    TOOLS_OF_THE_TRADE = 'tools of the trade'
    THUNDERCLAP = 'thunderclap'
    THUNDER_STRIKE = 'thunder strike'
    TRANQUILITY = 'tranquility'
    TRIP = 'trip'
    TURBO = 'turbo'
    TWIN_STRIKE = 'twin strike'
    UNLOAD = 'unload'
    UPPERCUT = 'uppercut'
    VIGILANCE = 'vigilance'
    VOID = 'void'
    WHIRLWIND = 'whirlwind'
    WORSHIP = 'worship'
    WILD_STRIKE = 'wild strike'
    WOUND = 'wound'
    WRAITH_FORM = 'wraith form v2'  # Weird id alert!
    ZAP = 'zap'
