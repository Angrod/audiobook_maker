# Roadmap

The to-do list for this fork. Tick boxes (`- [x]`) as you go so the next session knows where you left off.

**Where things stand (Oct 2026):** everything below "Done" is on branch `claude/jolly-curie-dg9q4d` in both [Angrod/audiobook_maker](https://github.com/Angrod/audiobook_maker/tree/claude/jolly-curie-dg9q4d) and [Angrod/GPT-SoVITS-Package](https://github.com/Angrod/GPT-SoVITS-Package/tree/claude/jolly-curie-dg9q4d). Neither branch is merged yet, and nothing has run on a GPU or produced real speech yet.

## To do

### 1. Test on your PC
- [ ] `git fetch`, `git checkout claude/jolly-curie-dg9q4d`, `git pull`
- [ ] `git submodule sync`, then `git submodule update --init modules/GPT-SoVITS-Package`
- [ ] Activate the venv, then `pip install .\modules\GPT-SoVITS-Package\` and `pip install -r requirements.txt`
- [ ] `pip install faster-whisper`
- [ ] Check torch is still `2.7.0+cu128` (`pip show torch`); reinstall it if not (see [Check Torch Install](README.md#check-torch-install))
- [ ] Load a chapter as a **new** audiobook so it gets the new chunking
- [ ] Generate it with an existing v2/v4 voice
- [ ] Run **Tools → Check Audio with Whisper**
- [ ] Set Seed to -1, run **Regenerate in Bulk**, then check again
- [ ] Note any errors (console output) and whether the flagged clips were real mistakes

### 2. Tune settings
- [ ] Read `qa_report.txt` in the audiobook folder: are the flags real mistakes or false alarms?
- [ ] Hallucinations still getting through: lower `chunk_max_chars` (try 180) or raise Repetition Penalty
- [ ] Too many false alarms: raise `qa_max_error_rate` or use a bigger `qa_whisper_model` (`medium`)
- [ ] Optional: try `export_format: m4b` and `export_normalize_loudness: true`

Settings are in `configs/settings.yaml`; see the table in the [README](README.md#settings-configssettingsyaml).

### 3. Merge
- [ ] `GPT-SoVITS-Package`: merge `claude/jolly-curie-dg9q4d` into `main` first
- [ ] `audiobook_maker`: merge `claude/jolly-curie-dg9q4d` into `master`
- [ ] Don't delete the package branch until it's merged: the app's submodule points at a commit on it

### 4. Try a v2ProPlus voice
- [ ] Build a dataset with JarodMica's [dataset-maker](https://github.com/JarodMica/dataset-maker) (GPT-SoVITS export)
- [ ] Train in the official [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) with v2ProPlus selected
- [ ] Upload the new models in the app and pick v2ProPlus under "Version to Use"
- [ ] Generate the same chapter with the old voice and the new one, and compare

### 5. Later (roughly in order of payoff)
- [ ] **Commit the automation pipeline.** The README mentions it, but it isn't in the repo yet
- [ ] **Automatic fix loop.** Check, regenerate and re-check until clean or N attempts
- [ ] **EPUB/PDF import.** Top of `bugs_and_feature_tracker.md`
- [ ] **Chapters in m4b exports.** So players can show and skip chapters
- [ ] **Qwen3-TTS as a second engine.** To compare against GPT-SoVITS on the same text
- [ ] **GPT-SoVITS v5 port.** Released upstream Oct 6, 2026 and builds on v2Pro; wait a few weeks for upstream fixes

## Done
- [x] Sentence-based chunking with a length cap, per-book settings, Tools → Text Chunking Settings (`a185a6a`)
- [x] GPT-SoVITS: Speed slider fixed, Repetition Penalty, Internal Text Split, extra reference clips (`e8cca8b`)
- [x] Tools → Check Audio with Whisper, `qa_report.txt`, unreadable clips flagged (`e78a0d0`, `58f2a6b`)
- [x] Export: skips missing audio, handles mixed sample rates, m4b and loudness options, runs in the background (`b9b865a`)
- [x] Copied GPT-SoVITS-Package to Angrod and pointed the submodule at it (`30174f6`)
- [x] v2Pro / v2ProPlus ported into GPT-SoVITS-Package (`6a0fe65`) and enabled in the app (`6304011`)
