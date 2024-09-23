# Bottled AI
Customizable Slay the Spire bot

Also check out our [YouTube channel](https://www.youtube.com/@BottledAI)!

## Setup

### Python Setup
1) Have Python 3.11.8 or so installed
    - Windows: you will likely need to add Python to the Path Environmental Variable
    - MacOS: you need Python 3.11+ within xcode. This requires xcode 14.0+ (this applied for python 3.9+, might need to be higher now), which in turn requires MacOS Monterey (lower versions won't work!)
2) Have [PIP](https://pip.pypa.io/en/stable/installation/) (python package manager) installed.

### Project Setup
1) Clone this repository into the game's install folder, in a new folder: `ai\requested_strike`.
    - Windows example: ` E:\Steam\steamapps\common\SlayTheSpire\ai\requested_strike`
    - MacOS example: Browse local files of StS via Steam -> Right click and Show Package Contents -> Resources -> ai -> requested_strike
2) Run `python -m pip install -r requirements.txt` in the root folder of the project (installs python dependencies)

### Game Setup
1) Through the steam workshop, make sure you have:
    - [BaseMod](https://steamcommunity.com/sharedfiles/filedetails/?id=1605833019) 
    - [StSLib](https://steamcommunity.com/sharedfiles/filedetails/?id=1609158507)
    - [ModTheSpire](https://steamcommunity.com/sharedfiles/filedetails/?id=1605060445)
    - [Communication Mod](https://steamcommunity.com/sharedfiles/filedetails/?id=2131373661)
2) Run the game with the mods enabled
3) TODO: reword 3+4. Now your spire config file should be created! See https://github.com/kiooeht/ModTheSpire/wiki/SpireConfig for how to navigate there
4) Find `config.properties` in the CommunicationMod folder. Then:
    - Windows: Add `command=python .\\ai\\requested_strike\\main.py` to the `config.properties` file there.
    - MacOS: Add `command=python3 ./ai/requested_strike/main.py` to the `config.properties` file there.

### Running the bot
- Run the bot via the game's main menu:
  - Mods ->
  - Communication Mod ->
  - Config (next to "Return")  ->
  - Start external process
- You can configure some run settings in [main.py](main.py).

Now it should all be able to run!

The process has a timeout of 10s so if you simply see that delay but nothing's happening, then something isn't working.
To debug, check the output in the ModTheSpire console, or the `communication_mod_errors.log` in the StS folder.

## Making your own bot
- See [how_to_make_your_own_bot.md](how_to_make_your_own_bot.md).


## Contact
Just raise an issue here on Github and we'll reach out!


## Tools

### Bot Controls
- Adjust which bot strategy is used, the amount of runs, and the seed in [main.py](main.py).
- Pause the bot in [run_controller.txt](run_controller.txt).
- Adjust the speed of certain actions in [presentation_config.py](presentation_config.py).

### Tests
- All tests can be found in the `/tests` directory
- VERY useful for checking bot behavior without needing to run the game
- You can run coverage checks with:
  - `python -m coverage run -m unittest discover .\tests`
  - `python -m coverage report`
  - `python -m coverage html`