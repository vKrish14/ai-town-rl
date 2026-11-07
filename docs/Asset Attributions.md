# Asset attributions

The viewer's map illustration, buildings, trees, gardens, lamps, and citizen sprites are original procedural pixel-inspired drawings authored in `ai_town/visuals.py`. No third-party image, sprite, or audio assets are used. The application uses the operating system's Segoe UI font when available, with Pygame's font fallback.

Floating wellbeing bars are drawn at runtime from current citizen state by `ai_town.visuals.draw_wellbeing_bar`; their colors are interpolated across the red–orange–yellow–green score ramp. They are UI indicators, not imported art assets.

Record the creator, source URL, license, and any required attribution here before adding external visual or audio assets.
