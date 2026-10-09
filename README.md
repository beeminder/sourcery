## Sourcery

This is a utility to compile a document with all the exact prompts you used to build whatever thing you've vibe-coded.
It gives you a way to peruse the whole dialog with the AI that spawned/effectuated the app.

Only the human-written words are default-visible.
You can click-to-expand to see the AI replies you're curious about. 

Part of the idea is that this the closest analog to actual source code that a vibe-coded app has, so it seems important to capture it.

## Usage

`sourcery.py REPO_DIRECTORY OUTPUT.html [--open]`

## Sourcery Dogfood

See how this tool itself came to life at [sourcery.html][sourcery.html].

## Multiple Humans

It should work now with any number of humans talking to any number of coding agents.
Run sourcery.py and it saves all your transcripts into a JSON file for you, named with your username, and merges it into a master sourcery.html which has everyone's transcripts.

## Up Next

* [TIM] Hovering over a time of day should show the full date/time with timezone
* [CON] One of the mockups Claude showed had a summary of contributions from each human. Let's get that implemented.
* [FAV] Better favicon. A wizard hat or magic wand or something. Or that overlaid on some ones and zeros, as in source code.
* [WHO] Check what happens if you don't have gh installed, which it might need for getting your GitHub username. Maybe fall back to `whoami`?


## Other name ideas and scratch notes

`appocrypha`
`codexegesis`

[sourcery.html]: https://beeminder.github.io/sourcery/sourcery.html