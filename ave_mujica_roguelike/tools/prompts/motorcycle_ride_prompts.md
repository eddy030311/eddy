# 夜行机车 · 骑乘动画提示词（直接复制）

每一段都是完整的一条提示词，整段复制贴进生图软件即可。

## 怎么做
1. **先生成 V0（单独的机车）**，挑一张满意的。之后每个角色的图都把 V0 一起上传当参考图，这样 8 个人骑的是同一辆车。
2. **每个角色再上传她现在的游戏精灵图**当角色参考（我一并发给你了：`doloris.png`、`mortis.png` 等），人物比例和服装才会和游戏里一致。便服三人也可以用当初的设定资料图。
3. 背景统一纯品红 `#FF00FF`（我靠这个颜色抠图）；能直接导出透明 PNG 也可以。
4. 发回时注明编号（V0、R1……R8）。帧数多一两帧、少一两帧都没关系，告诉我哪一格是什么动作就行。
5. Midjourney 用户在末尾加 `--style raw`，以及比例：V0 用 `--ar 24:5`，R1–R8 用 `--ar 12:5`。

## 每张图的格子（R1–R8）
- 4 列 × 2 行，每格 96×80 像素（整张 384×160，或等比放大，例如 1536×640）。
- 第 1 行：第 1–2 格 **停车坐着**；第 3–4 格 **撞击**（被弹得往后仰）。
- 第 2 行：4 格 **行驶循环**（前倾、头发和裙摆往后飘、轮辐转动、车灯亮）。
- 人和车都面朝右，我在游戏里镜像。

## 负面提示词（所有图通用，软件支持的话填进 Negative prompt）
```
blurry, anti-aliasing, smooth shading, painterly, 3d render, photo, realistic, gradient background, white background, transparent checkerboard pattern, floor, road, scenery, text, letters, watermark, signature, grid lines, labels, frame numbers, cropped, cut off, extra limbs, deformed hands, missing legs, floating rider, standing next to the motorcycle, pedalling, bicycle, scooter, modern sport bike, front view, three-quarter view, motion blur, speed lines, inconsistent motorcycle, different wheel sizes
```

---
## V0 · 夜行机车（单独，无人）　文件名：veh_bike.png
```
Pixel art game asset in crisp 16-bit SNES/GBA style: hard pixel edges, no anti-aliasing, no blur, no smooth gradients, limited palette of about 32 colors, clean 1-pixel dark outline (#120c10), soft cel shading with 2 to 3 tones per material. Theme: gothic Victorian masquerade theatre, deep wine red #9b1b30, antique gold #c9a45c and bone white #ece2d2 accents. Background: flat solid pure magenta #FF00FF everywhere outside the subject, no floor, no road, no cast shadow on the background, no gradient. No text, no letters, no numbers, no watermark, no signature, no border around the image.

The motorcycle (identical in every frame and in every sheet): a vintage gothic cafe-racer motorcycle seen exactly from the side, facing RIGHT; glossy black frame and engine block with three silver cooling fins; a deep wine-red #9b1b30 teardrop fuel tank with a small antique-gold rose emblem on its side; a short black leather seat behind the tank; spoked wheels with antique-gold rims and black tyres, both wheels the same size; a round brass lantern headlamp with a warm cream glass at the front; thin curved gold handlebars; a chrome exhaust pipe along the lower back; a black front fender with one tiny red rose; small silver footpegs below the seat.

Strict sprite sheet: an exact grid of 4 columns by 1 row of equal cells (96x80 game pixels per cell, whole sheet 384x80 pixels, or an exact integer upscale such as 1536x320). The motorcycle alone with nobody on it, about 64 pixels long with wheels about 22 pixels across, standing on the same baseline 4 pixels above the bottom edge of every cell, centered horizontally, side view facing RIGHT. Nothing crosses a cell edge. No grid lines, no labels, no rider, no effects.

Cell 1 — PARKED: standing still on its side stand, headlamp off.
Cells 2-4 — WHEELS TURNING, 3 frames: the same motorcycle upright with the headlamp lit; the wheel spokes rotate one third of a step in each frame so the three frames loop.
```

## R1 · Doloris · 三角初华（舞台装） · 骑车　文件名：ride_doloris.png
参考图：上传 V0 和 `doloris.png`。
```
Pixel art game asset in crisp 16-bit SNES/GBA style: hard pixel edges, no anti-aliasing, no blur, no smooth gradients, limited palette of about 32 colors, clean 1-pixel dark outline (#120c10), soft cel shading with 2 to 3 tones per material. Theme: gothic Victorian masquerade theatre, deep wine red #9b1b30, antique gold #c9a45c and bone white #ece2d2 accents. Background: flat solid pure magenta #FF00FF everywhere outside the subject, no floor, no road, no cast shadow on the background, no gradient. No text, no letters, no numbers, no watermark, no signature, no border around the image.

Sprite sheet of Doloris (Uika Misumi in her stage costume), a girl with shoulder-length straight platinum-blonde hair fading to rose pink at the tips, violet eyes and a black lace masquerade mask over her eyes. Outfit: a black off-shoulder gothic dress with a fitted bodice, a layered ruffled skirt fading from black to magenta-pink, black thigh-high stockings with garter straps, black lace-up boots. Instrument: a black electric guitar with a wooden neck, slung on her back.

The motorcycle (identical in every frame and in every sheet): a vintage gothic cafe-racer motorcycle seen exactly from the side, facing RIGHT; glossy black frame and engine block with three silver cooling fins; a deep wine-red #9b1b30 teardrop fuel tank with a small antique-gold rose emblem on its side; a short black leather seat behind the tank; spoked wheels with antique-gold rims and black tyres, both wheels the same size; a round brass lantern headlamp with a warm cream glass at the front; thin curved gold handlebars; a chrome exhaust pipe along the lower back; a black front fender with one tiny red rose; small silver footpegs below the seat.

Strict sprite sheet: an exact grid of 4 columns by 2 rows of equal cells (96x80 game pixels per cell, whole sheet 384x160 pixels, or an exact integer upscale such as 1536x640). Side view; rider and motorcycle face RIGHT in every frame. The motorcycle is about 64 pixels long with wheels about 22 pixels across, standing on the same baseline 4 pixels above the bottom edge of every cell, centered horizontally. The rider has exactly the same chibi proportions and pixel size as the attached reference sprite sheet (about 2.5 heads tall, about 44 pixels tall when standing), now SEATED astride the motorcycle: hips on the seat, the near leg bent with the thigh along the side of the fuel tank and the boot resting on the footpeg, both hands on the handlebars, the instrument slung on her back by its strap. The legs must be fully visible and connected to the body; the rider never floats above the seat. The same character, outfit, motorcycle, scale and lighting in every frame. Nothing crosses a cell edge. No grid lines, no labels, no motion lines, no speed lines, no dust and no effects: only the rider and the motorcycle. Unused cells stay empty background.

Row 1, cells 1-2 — PARKED, 2 frames: sitting upright on the stopped motorcycle, hands resting on the handlebars, gentle breathing, hair and skirt hanging still, headlamp off.
Row 1, cells 3-4 — CRASH, 2 frames: the motorcycle jolts as it hits something; the rider is thrown back on the seat with eyes shut, hair and skirt flung forward, one hand lifting off the handlebar, then she grips it again.
Row 2 — RIDING, 4 frames: a looping riding cycle at speed; leaning forward over the tank, chin down, hair, ribbons, coat-tails and skirt streaming backward and rippling a little differently in each frame; the wheel spokes rotate a quarter step each frame; headlamp lit; the rider bobs up and down by 1 pixel.
```

## R2 · Mortis · 若叶睦（舞台装） · 骑车　文件名：ride_mortis.png
参考图：上传 V0 和 `mortis.png`。
```
Pixel art game asset in crisp 16-bit SNES/GBA style: hard pixel edges, no anti-aliasing, no blur, no smooth gradients, limited palette of about 32 colors, clean 1-pixel dark outline (#120c10), soft cel shading with 2 to 3 tones per material. Theme: gothic Victorian masquerade theatre, deep wine red #9b1b30, antique gold #c9a45c and bone white #ece2d2 accents. Background: flat solid pure magenta #FF00FF everywhere outside the subject, no floor, no road, no cast shadow on the background, no gradient. No text, no letters, no numbers, no watermark, no signature, no border around the image.

Sprite sheet of Mortis (Mutsumi Wakaba in her stage costume), a girl with very long straight pale sage-silver hair, red eyes, a crimson beret and a black cloth mask covering her nose and mouth. Outfit: a dark burgundy-brown gothic dress with a long tailcoat-style back, a green brooch at the collar, dark tights and crimson shoes. Instrument: a hot-pink electric guitar, slung on her back.

The motorcycle (identical in every frame and in every sheet): a vintage gothic cafe-racer motorcycle seen exactly from the side, facing RIGHT; glossy black frame and engine block with three silver cooling fins; a deep wine-red #9b1b30 teardrop fuel tank with a small antique-gold rose emblem on its side; a short black leather seat behind the tank; spoked wheels with antique-gold rims and black tyres, both wheels the same size; a round brass lantern headlamp with a warm cream glass at the front; thin curved gold handlebars; a chrome exhaust pipe along the lower back; a black front fender with one tiny red rose; small silver footpegs below the seat.

Strict sprite sheet: an exact grid of 4 columns by 2 rows of equal cells (96x80 game pixels per cell, whole sheet 384x160 pixels, or an exact integer upscale such as 1536x640). Side view; rider and motorcycle face RIGHT in every frame. The motorcycle is about 64 pixels long with wheels about 22 pixels across, standing on the same baseline 4 pixels above the bottom edge of every cell, centered horizontally. The rider has exactly the same chibi proportions and pixel size as the attached reference sprite sheet (about 2.5 heads tall, about 44 pixels tall when standing), now SEATED astride the motorcycle: hips on the seat, the near leg bent with the thigh along the side of the fuel tank and the boot resting on the footpeg, both hands on the handlebars, the instrument slung on her back by its strap. The legs must be fully visible and connected to the body; the rider never floats above the seat. The same character, outfit, motorcycle, scale and lighting in every frame. Nothing crosses a cell edge. No grid lines, no labels, no motion lines, no speed lines, no dust and no effects: only the rider and the motorcycle. Unused cells stay empty background.

Row 1, cells 1-2 — PARKED, 2 frames: sitting upright on the stopped motorcycle, hands resting on the handlebars, gentle breathing, hair and skirt hanging still, headlamp off.
Row 1, cells 3-4 — CRASH, 2 frames: the motorcycle jolts as it hits something; the rider is thrown back on the seat with eyes shut, hair and skirt flung forward, one hand lifting off the handlebar, then she grips it again.
Row 2 — RIDING, 4 frames: a looping riding cycle at speed; leaning forward over the tank, chin down, hair, ribbons, coat-tails and skirt streaming backward and rippling a little differently in each frame; the wheel spokes rotate a quarter step each frame; headlamp lit; the rider bobs up and down by 1 pixel.
```

## R3 · Timoris · 八幡海铃（舞台装） · 骑车　文件名：ride_timoris.png
参考图：上传 V0 和 `timoris.png`。
```
Pixel art game asset in crisp 16-bit SNES/GBA style: hard pixel edges, no anti-aliasing, no blur, no smooth gradients, limited palette of about 32 colors, clean 1-pixel dark outline (#120c10), soft cel shading with 2 to 3 tones per material. Theme: gothic Victorian masquerade theatre, deep wine red #9b1b30, antique gold #c9a45c and bone white #ece2d2 accents. Background: flat solid pure magenta #FF00FF everywhere outside the subject, no floor, no road, no cast shadow on the background, no gradient. No text, no letters, no numbers, no watermark, no signature, no border around the image.

Sprite sheet of Timoris (Umiri Yahata in her stage costume), a tall cool girl with very long straight dark violet-black hair, teal eyes and a black cloth mask covering her nose and mouth. Outfit: a short dark-red capelet over the shoulders, a black fitted top, black shorts, black thigh-high boots. Instrument: a pale lilac electric bass with a purple flower ornament on the headstock, slung on her back.

The motorcycle (identical in every frame and in every sheet): a vintage gothic cafe-racer motorcycle seen exactly from the side, facing RIGHT; glossy black frame and engine block with three silver cooling fins; a deep wine-red #9b1b30 teardrop fuel tank with a small antique-gold rose emblem on its side; a short black leather seat behind the tank; spoked wheels with antique-gold rims and black tyres, both wheels the same size; a round brass lantern headlamp with a warm cream glass at the front; thin curved gold handlebars; a chrome exhaust pipe along the lower back; a black front fender with one tiny red rose; small silver footpegs below the seat.

Strict sprite sheet: an exact grid of 4 columns by 2 rows of equal cells (96x80 game pixels per cell, whole sheet 384x160 pixels, or an exact integer upscale such as 1536x640). Side view; rider and motorcycle face RIGHT in every frame. The motorcycle is about 64 pixels long with wheels about 22 pixels across, standing on the same baseline 4 pixels above the bottom edge of every cell, centered horizontally. The rider has exactly the same chibi proportions and pixel size as the attached reference sprite sheet (about 2.5 heads tall, about 44 pixels tall when standing), now SEATED astride the motorcycle: hips on the seat, the near leg bent with the thigh along the side of the fuel tank and the boot resting on the footpeg, both hands on the handlebars, the instrument slung on her back by its strap. The legs must be fully visible and connected to the body; the rider never floats above the seat. The same character, outfit, motorcycle, scale and lighting in every frame. Nothing crosses a cell edge. No grid lines, no labels, no motion lines, no speed lines, no dust and no effects: only the rider and the motorcycle. Unused cells stay empty background.

Row 1, cells 1-2 — PARKED, 2 frames: sitting upright on the stopped motorcycle, hands resting on the handlebars, gentle breathing, hair and skirt hanging still, headlamp off.
Row 1, cells 3-4 — CRASH, 2 frames: the motorcycle jolts as it hits something; the rider is thrown back on the seat with eyes shut, hair and skirt flung forward, one hand lifting off the handlebar, then she grips it again.
Row 2 — RIDING, 4 frames: a looping riding cycle at speed; leaning forward over the tank, chin down, hair, ribbons, coat-tails and skirt streaming backward and rippling a little differently in each frame; the wheel spokes rotate a quarter step each frame; headlamp lit; the rider bobs up and down by 1 pixel.
```

## R4 · Amoris · 祐天寺若麦（舞台装） · 骑车　文件名：ride_amoris.png
参考图：上传 V0 和 `amoris.png`。
```
Pixel art game asset in crisp 16-bit SNES/GBA style: hard pixel edges, no anti-aliasing, no blur, no smooth gradients, limited palette of about 32 colors, clean 1-pixel dark outline (#120c10), soft cel shading with 2 to 3 tones per material. Theme: gothic Victorian masquerade theatre, deep wine red #9b1b30, antique gold #c9a45c and bone white #ece2d2 accents. Background: flat solid pure magenta #FF00FF everywhere outside the subject, no floor, no road, no cast shadow on the background, no gradient. No text, no letters, no numbers, no watermark, no signature, no border around the image.

Sprite sheet of Amoris (Nyamu Yutenji in her stage costume), a girl with short wavy purple-magenta hair, red eyes and a black masquerade mask over her eyes. Outfit: a dark brown cropped jacket over a black top with a silver chain necklace, a short red pleated skirt, crimson thigh-high stockings, red heels. Held item: a pair of black drumsticks tucked into a holster on her back (no drum kit).

The motorcycle (identical in every frame and in every sheet): a vintage gothic cafe-racer motorcycle seen exactly from the side, facing RIGHT; glossy black frame and engine block with three silver cooling fins; a deep wine-red #9b1b30 teardrop fuel tank with a small antique-gold rose emblem on its side; a short black leather seat behind the tank; spoked wheels with antique-gold rims and black tyres, both wheels the same size; a round brass lantern headlamp with a warm cream glass at the front; thin curved gold handlebars; a chrome exhaust pipe along the lower back; a black front fender with one tiny red rose; small silver footpegs below the seat.

Strict sprite sheet: an exact grid of 4 columns by 2 rows of equal cells (96x80 game pixels per cell, whole sheet 384x160 pixels, or an exact integer upscale such as 1536x640). Side view; rider and motorcycle face RIGHT in every frame. The motorcycle is about 64 pixels long with wheels about 22 pixels across, standing on the same baseline 4 pixels above the bottom edge of every cell, centered horizontally. The rider has exactly the same chibi proportions and pixel size as the attached reference sprite sheet (about 2.5 heads tall, about 44 pixels tall when standing), now SEATED astride the motorcycle: hips on the seat, the near leg bent with the thigh along the side of the fuel tank and the boot resting on the footpeg, both hands on the handlebars, the instrument slung on her back by its strap. The legs must be fully visible and connected to the body; the rider never floats above the seat. The same character, outfit, motorcycle, scale and lighting in every frame. Nothing crosses a cell edge. No grid lines, no labels, no motion lines, no speed lines, no dust and no effects: only the rider and the motorcycle. Unused cells stay empty background.

Row 1, cells 1-2 — PARKED, 2 frames: sitting upright on the stopped motorcycle, hands resting on the handlebars, gentle breathing, hair and skirt hanging still, headlamp off.
Row 1, cells 3-4 — CRASH, 2 frames: the motorcycle jolts as it hits something; the rider is thrown back on the seat with eyes shut, hair and skirt flung forward, one hand lifting off the handlebar, then she grips it again.
Row 2 — RIDING, 4 frames: a looping riding cycle at speed; leaning forward over the tank, chin down, hair, ribbons, coat-tails and skirt streaming backward and rippling a little differently in each frame; the wheel spokes rotate a quarter step each frame; headlamp lit; the rider bobs up and down by 1 pixel.
```

## R5 · Oblivionis · 丰川祥子（舞台装） · 骑车　文件名：ride_oblivionis.png
参考图：上传 V0 和 `oblivionis.png`。
```
Pixel art game asset in crisp 16-bit SNES/GBA style: hard pixel edges, no anti-aliasing, no blur, no smooth gradients, limited palette of about 32 colors, clean 1-pixel dark outline (#120c10), soft cel shading with 2 to 3 tones per material. Theme: gothic Victorian masquerade theatre, deep wine red #9b1b30, antique gold #c9a45c and bone white #ece2d2 accents. Background: flat solid pure magenta #FF00FF everywhere outside the subject, no floor, no road, no cast shadow on the background, no gradient. No text, no letters, no numbers, no watermark, no signature, no border around the image.

Sprite sheet of Oblivionis (Sakiko Togawa in her stage costume), a girl with long wavy lavender-blue hair tied with black ribbons, amber eyes and a black masquerade mask over her eyes. Outfit: a red and black gothic dress with a high collar, a dark plaid layered skirt, black knee socks and black boots. Instrument: none (her keyboard stays at home); a small black ribbon flutters from her hair.

The motorcycle (identical in every frame and in every sheet): a vintage gothic cafe-racer motorcycle seen exactly from the side, facing RIGHT; glossy black frame and engine block with three silver cooling fins; a deep wine-red #9b1b30 teardrop fuel tank with a small antique-gold rose emblem on its side; a short black leather seat behind the tank; spoked wheels with antique-gold rims and black tyres, both wheels the same size; a round brass lantern headlamp with a warm cream glass at the front; thin curved gold handlebars; a chrome exhaust pipe along the lower back; a black front fender with one tiny red rose; small silver footpegs below the seat.

Strict sprite sheet: an exact grid of 4 columns by 2 rows of equal cells (96x80 game pixels per cell, whole sheet 384x160 pixels, or an exact integer upscale such as 1536x640). Side view; rider and motorcycle face RIGHT in every frame. The motorcycle is about 64 pixels long with wheels about 22 pixels across, standing on the same baseline 4 pixels above the bottom edge of every cell, centered horizontally. The rider has exactly the same chibi proportions and pixel size as the attached reference sprite sheet (about 2.5 heads tall, about 44 pixels tall when standing), now SEATED astride the motorcycle: hips on the seat, the near leg bent with the thigh along the side of the fuel tank and the boot resting on the footpeg, both hands on the handlebars, the instrument slung on her back by its strap. The legs must be fully visible and connected to the body; the rider never floats above the seat. The same character, outfit, motorcycle, scale and lighting in every frame. Nothing crosses a cell edge. No grid lines, no labels, no motion lines, no speed lines, no dust and no effects: only the rider and the motorcycle. Unused cells stay empty background.

Row 1, cells 1-2 — PARKED, 2 frames: sitting upright on the stopped motorcycle, hands resting on the handlebars, gentle breathing, hair and skirt hanging still, headlamp off.
Row 1, cells 3-4 — CRASH, 2 frames: the motorcycle jolts as it hits something; the rider is thrown back on the seat with eyes shut, hair and skirt flung forward, one hand lifting off the handlebar, then she grips it again.
Row 2 — RIDING, 4 frames: a looping riding cycle at speed; leaning forward over the tank, chin down, hair, ribbons, coat-tails and skirt streaming backward and rippling a little differently in each frame; the wheel spokes rotate a quarter step each frame; headlamp lit; the rider bobs up and down by 1 pixel.
```

## R6 · 三角初华（便服） · 骑车　文件名：ride_uika.png
参考图：上传 V0 和 `uika.png`。
```
Pixel art game asset in crisp 16-bit SNES/GBA style: hard pixel edges, no anti-aliasing, no blur, no smooth gradients, limited palette of about 32 colors, clean 1-pixel dark outline (#120c10), soft cel shading with 2 to 3 tones per material. Theme: gothic Victorian masquerade theatre, deep wine red #9b1b30, antique gold #c9a45c and bone white #ece2d2 accents. Background: flat solid pure magenta #FF00FF everywhere outside the subject, no floor, no road, no cast shadow on the background, no gradient. No text, no letters, no numbers, no watermark, no signature, no border around the image.

Sprite sheet of Uika Misumi in casual clothes, a girl with shoulder-length straight platinum-blonde hair fading to soft pink at the tips, side-swept bangs, violet eyes and a gentle smile. Outfit: a black baseball cap, a plain white short-sleeve T-shirt tucked into a long beige A-line denim maxi skirt with front pockets, white ankle socks, black canvas high-top sneakers with white soles. Instrument: a wooden acoustic guitar on a black strap, slung on her back.

The motorcycle (identical in every frame and in every sheet): a vintage gothic cafe-racer motorcycle seen exactly from the side, facing RIGHT; glossy black frame and engine block with three silver cooling fins; a deep wine-red #9b1b30 teardrop fuel tank with a small antique-gold rose emblem on its side; a short black leather seat behind the tank; spoked wheels with antique-gold rims and black tyres, both wheels the same size; a round brass lantern headlamp with a warm cream glass at the front; thin curved gold handlebars; a chrome exhaust pipe along the lower back; a black front fender with one tiny red rose; small silver footpegs below the seat.

Strict sprite sheet: an exact grid of 4 columns by 2 rows of equal cells (96x80 game pixels per cell, whole sheet 384x160 pixels, or an exact integer upscale such as 1536x640). Side view; rider and motorcycle face RIGHT in every frame. The motorcycle is about 64 pixels long with wheels about 22 pixels across, standing on the same baseline 4 pixels above the bottom edge of every cell, centered horizontally. The rider has exactly the same chibi proportions and pixel size as the attached reference sprite sheet (about 2.5 heads tall, about 44 pixels tall when standing), now SEATED astride the motorcycle: hips on the seat, the near leg bent with the thigh along the side of the fuel tank and the boot resting on the footpeg, both hands on the handlebars, the instrument slung on her back by its strap. The legs must be fully visible and connected to the body; the rider never floats above the seat. The same character, outfit, motorcycle, scale and lighting in every frame. Nothing crosses a cell edge. No grid lines, no labels, no motion lines, no speed lines, no dust and no effects: only the rider and the motorcycle. Unused cells stay empty background.

Row 1, cells 1-2 — PARKED, 2 frames: sitting upright on the stopped motorcycle, hands resting on the handlebars, gentle breathing, hair and skirt hanging still, headlamp off.
Row 1, cells 3-4 — CRASH, 2 frames: the motorcycle jolts as it hits something; the rider is thrown back on the seat with eyes shut, hair and skirt flung forward, one hand lifting off the handlebar, then she grips it again.
Row 2 — RIDING, 4 frames: a looping riding cycle at speed; leaning forward over the tank, chin down, hair, ribbons, coat-tails and skirt streaming backward and rippling a little differently in each frame; the wheel spokes rotate a quarter step each frame; headlamp lit; the rider bobs up and down by 1 pixel.
```

## R7 · 若叶睦（便服） · 骑车　文件名：ride_mutsumi.png
参考图：上传 V0 和 `mutsumi.png`。
```
Pixel art game asset in crisp 16-bit SNES/GBA style: hard pixel edges, no anti-aliasing, no blur, no smooth gradients, limited palette of about 32 colors, clean 1-pixel dark outline (#120c10), soft cel shading with 2 to 3 tones per material. Theme: gothic Victorian masquerade theatre, deep wine red #9b1b30, antique gold #c9a45c and bone white #ece2d2 accents. Background: flat solid pure magenta #FF00FF everywhere outside the subject, no floor, no road, no cast shadow on the background, no gradient. No text, no letters, no numbers, no watermark, no signature, no border around the image.

Sprite sheet of Mutsumi Wakaba in casual clothes, a girl with very long straight pale sage-silver hair past the waist with blunt bangs, amber eyes and a calm, nearly expressionless face. Outfit: a black velvet choker, a navy-blue tartan plaid long-sleeve dress with a ruffled hem just below the knee, a cream lace bib collar with a small navy ribbon bow, a belt at the waist, cream cuffs with buttons, white frilled ankle socks, dark brown Mary-Jane heels. Instrument: a pale cream electric guitar, slung on her back.

The motorcycle (identical in every frame and in every sheet): a vintage gothic cafe-racer motorcycle seen exactly from the side, facing RIGHT; glossy black frame and engine block with three silver cooling fins; a deep wine-red #9b1b30 teardrop fuel tank with a small antique-gold rose emblem on its side; a short black leather seat behind the tank; spoked wheels with antique-gold rims and black tyres, both wheels the same size; a round brass lantern headlamp with a warm cream glass at the front; thin curved gold handlebars; a chrome exhaust pipe along the lower back; a black front fender with one tiny red rose; small silver footpegs below the seat.

Strict sprite sheet: an exact grid of 4 columns by 2 rows of equal cells (96x80 game pixels per cell, whole sheet 384x160 pixels, or an exact integer upscale such as 1536x640). Side view; rider and motorcycle face RIGHT in every frame. The motorcycle is about 64 pixels long with wheels about 22 pixels across, standing on the same baseline 4 pixels above the bottom edge of every cell, centered horizontally. The rider has exactly the same chibi proportions and pixel size as the attached reference sprite sheet (about 2.5 heads tall, about 44 pixels tall when standing), now SEATED astride the motorcycle: hips on the seat, the near leg bent with the thigh along the side of the fuel tank and the boot resting on the footpeg, both hands on the handlebars, the instrument slung on her back by its strap. The legs must be fully visible and connected to the body; the rider never floats above the seat. The same character, outfit, motorcycle, scale and lighting in every frame. Nothing crosses a cell edge. No grid lines, no labels, no motion lines, no speed lines, no dust and no effects: only the rider and the motorcycle. Unused cells stay empty background.

Row 1, cells 1-2 — PARKED, 2 frames: sitting upright on the stopped motorcycle, hands resting on the handlebars, gentle breathing, hair and skirt hanging still, headlamp off.
Row 1, cells 3-4 — CRASH, 2 frames: the motorcycle jolts as it hits something; the rider is thrown back on the seat with eyes shut, hair and skirt flung forward, one hand lifting off the handlebar, then she grips it again.
Row 2 — RIDING, 4 frames: a looping riding cycle at speed; leaning forward over the tank, chin down, hair, ribbons, coat-tails and skirt streaming backward and rippling a little differently in each frame; the wheel spokes rotate a quarter step each frame; headlamp lit; the rider bobs up and down by 1 pixel.
```

## R8 · 丰川祥子（便服） · 骑车　文件名：ride_sakiko.png
参考图：上传 V0 和 `sakiko.png`。
```
Pixel art game asset in crisp 16-bit SNES/GBA style: hard pixel edges, no anti-aliasing, no blur, no smooth gradients, limited palette of about 32 colors, clean 1-pixel dark outline (#120c10), soft cel shading with 2 to 3 tones per material. Theme: gothic Victorian masquerade theatre, deep wine red #9b1b30, antique gold #c9a45c and bone white #ece2d2 accents. Background: flat solid pure magenta #FF00FF everywhere outside the subject, no floor, no road, no cast shadow on the background, no gradient. No text, no letters, no numbers, no watermark, no signature, no border around the image.

Sprite sheet of Sakiko Togawa in casual clothes, a girl with long wavy blue-lavender hair with two small side braids tied with black ribbons, side-swept bangs, amber eyes and an elegant gentle smile. Outfit: a pale pink long-sleeve blouse with a pleated front, buttoned cuffs and a thin black ribbon bow tie, a high-waisted grey tartan plaid A-line midi skirt with a black waistband, white frilled ankle socks, black Mary-Jane shoes. Held item: a black leather sheet-music folder tucked into a small bag on her back.

The motorcycle (identical in every frame and in every sheet): a vintage gothic cafe-racer motorcycle seen exactly from the side, facing RIGHT; glossy black frame and engine block with three silver cooling fins; a deep wine-red #9b1b30 teardrop fuel tank with a small antique-gold rose emblem on its side; a short black leather seat behind the tank; spoked wheels with antique-gold rims and black tyres, both wheels the same size; a round brass lantern headlamp with a warm cream glass at the front; thin curved gold handlebars; a chrome exhaust pipe along the lower back; a black front fender with one tiny red rose; small silver footpegs below the seat.

Strict sprite sheet: an exact grid of 4 columns by 2 rows of equal cells (96x80 game pixels per cell, whole sheet 384x160 pixels, or an exact integer upscale such as 1536x640). Side view; rider and motorcycle face RIGHT in every frame. The motorcycle is about 64 pixels long with wheels about 22 pixels across, standing on the same baseline 4 pixels above the bottom edge of every cell, centered horizontally. The rider has exactly the same chibi proportions and pixel size as the attached reference sprite sheet (about 2.5 heads tall, about 44 pixels tall when standing), now SEATED astride the motorcycle: hips on the seat, the near leg bent with the thigh along the side of the fuel tank and the boot resting on the footpeg, both hands on the handlebars, the instrument slung on her back by its strap. The legs must be fully visible and connected to the body; the rider never floats above the seat. The same character, outfit, motorcycle, scale and lighting in every frame. Nothing crosses a cell edge. No grid lines, no labels, no motion lines, no speed lines, no dust and no effects: only the rider and the motorcycle. Unused cells stay empty background.

Row 1, cells 1-2 — PARKED, 2 frames: sitting upright on the stopped motorcycle, hands resting on the handlebars, gentle breathing, hair and skirt hanging still, headlamp off.
Row 1, cells 3-4 — CRASH, 2 frames: the motorcycle jolts as it hits something; the rider is thrown back on the seat with eyes shut, hair and skirt flung forward, one hand lifting off the handlebar, then she grips it again.
Row 2 — RIDING, 4 frames: a looping riding cycle at speed; leaning forward over the tank, chin down, hair, ribbons, coat-tails and skirt streaming backward and rippling a little differently in each frame; the wheel spokes rotate a quarter step each frame; headlamp lit; the rider bobs up and down by 1 pixel.
```
