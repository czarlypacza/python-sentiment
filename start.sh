#!/bin/bash

x-terminal-emulator -e /media/michal/dev/sentiment/python/myenv/bin/python /media/michal/dev/sentiment/python-sentiment/sentiment.py

x-terminal-emulator -e /media/michal/dev/sentiment/python/myenv/bin/python /media/michal/dev/sentiment/python-sentiment/scrapper.py

# Open first terminal and run the first command
#gnome-terminal -- bash -c '/media/michal/dev/sentiment/python/myenv/bin/python /media/michal/dev/sentiment/python-sentiment/sentiment.py; exec bash'

# Open second terminal and run the second command
#gnome-terminal -- bash -c '/media/michal/dev/sentiment/python/myenv/bin/python /media/michal/dev/sentiment/python-sentiment/scrapper.py; exec bash'

# Open third terminal and run the third command
#gnome-terminal -- bash -c 'HSA_OVERRIDE_GFX_VERSION=10.3.0 /media/michal/dev/sentiment/python/myenv/bin/python /media/michal/dev/sentiment/python-sentiment/sentiment_BERT.py; exec bash'

# Open fourth terminal and run the fourth command
#gnome-terminal -- bash -c 'pnpm run dev; exec bash'

