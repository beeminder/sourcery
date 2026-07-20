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

## Notes to self

request: make it easier to visually distinguish which agent the human is talking to. right now all we have is the down-popped click-to-expand line at the end of the prompt.

later: have this work for multiple people. maybe get all the agent transcript data as a JSON file for each person, commit them all to version control, and then this utility makes sure the JSON file for the person running it is up to date and then generates the sourcery.html file from the set of JSON files?

sooner: speaking of JSON files, if you switch computers or something, you don't want to lose any of the transcript.

request: better favicon.

---

## Other name ideas and scratch notes

`appocrypha`
`codexegesis`

[sourcery.html]: https://beeminder.github.io/sourcery/sourcery.html