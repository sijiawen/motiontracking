# Creatorgen: video content-copy detection (MVP)

Upload an original video (A) and a suspected copy (B). Three independent tests check whether B reused A's content, and the **final score is the highest of the three**:

| Test | What it catches | How |
|---|---|---|
| Body & motion | Face swap over the same performance | 5 Sapiens2 models (pose, body parts, normals, pointmap, matting) |
| Scene & visuals | Same room, set and shot with someone else edited in | DINOv2 image fingerprints, with and without the person |
| Script | Same words read by a new face or AI voice | Whisper speech-to-text + shared 4+ word phrases |

```
Browser (Vercel) ──upload──▶ Supabase Storage (private "videos" bucket)
      │                              ▲
      │ create / analyze / status    │ download videos, upload result images
      ▼                              │
Modal web API ──spawn──▶ Modal GPU worker (H100, all 5 Sapiens2-1B models)
      │                              │
      └──────────▶ Supabase table "comparisons" ◀──┘   (every upload + result is logged here)
```

| Folder | What's in it |
|---|---|
| `web/` | The website. Plain HTML/CSS/JS, no build step. `admin.html` lists every upload. |
| `backend/modal_app.py` | GPU pipeline plus the small web API, deployed to Modal. |
| `backend/similarity.py` | Body & motion scoring: fingerprints, time alignment, mirror check. |
| `backend/content.py` | Scene and script tests, and how the three combine into the final score. |
| `backend/render.py` | Draws the example images for each model. |
| `backend/calibrate.py` | Tests the score on **your own labelled video set** and tunes the thresholds. |
| `supabase/schema.sql` | Database table and storage buckets. |

**Try the design first:** open a terminal in this folder and run `cd web && python3 -m http.server 8000`, then visit http://localhost:8000. With `API_URL` empty the site runs in demo mode with sample results.

---

## Setup (about 45 minutes, once)

You'll need Python 3.11+ (check with `python3 --version`) and a terminal. Commands below are run from this project folder.

### 1. Supabase (storage and database)

1. Create a free account at [supabase.com](https://supabase.com), then click **New project**. Pick any name and a strong database password, and choose the region closest to you (e.g. West US).
2. Once it's ready, go to **SQL Editor → New query**, paste the whole of `supabase/schema.sql`, and click **Run**. You should see "Success".
3. Go to **Project Settings → API** and copy three values into a note:
   - **Project URL**, e.g. `https://abcdxyz.supabase.co`
   - **anon public** key. This one goes in the website and is safe to be public.
   - **service_role** key. ⚠️ This one is secret. It only goes into Modal, never into `web/`.

### 2. Hugging Face (model weights)

Open [huggingface.co/facebook/sapiens2-pose-1b](https://huggingface.co/facebook/sapiens2-pose-1b).

- If you see an **"Agree and access repository"** box, the models are gated. Create a free account and click agree on each of the five model pages: pose, seg, normal, pointmap and matting (all `-1b`). Then create a **Read** token at *Settings → Access Tokens*.
- If there's no box, you can skip the token.

### 3. Modal (the GPU)

```bash
pip install modal numpy opencv-python-headless
modal setup                     # opens a browser to log in / sign up
```

Add a payment method in Modal's dashboard. Free credits go from $5 to $30/month, and it unlocks GPUs. Startups can also apply for credits on Modal's pricing page.

Store your secrets in Modal. Replace every `...` with your own values; `ADMIN_KEY` and `IP_SALT` can be any long random strings you make up:

```bash
modal secret create origin-secrets \
  SUPABASE_URL=https://abcdxyz.supabase.co \
  SUPABASE_SERVICE_ROLE_KEY=... \
  ADMIN_KEY=pick-a-long-password \
  IP_SALT=another-random-string \
  ALLOWED_ORIGINS="*" \
  HF_TOKEN=hf_...               # leave this line out if step 2 had no gate
```

Download the model weights (about 32 GB, 10–20 minutes, only once; re-running it later only fetches what's missing), then put everything online:

```bash
modal run backend/modal_app.py::download_models
modal deploy backend/modal_app.py
```

`deploy` prints a URL ending in `...-api.modal.run`. That's your **API URL**. Check it works by opening `<API URL>/health` in a browser.

### 4. Connect the website

Edit `web/config.js`:

```js
API_URL: "https://yourname--origin-video-compare-api.modal.run",
SUPABASE_URL: "https://abcdxyz.supabase.co",
SUPABASE_ANON_KEY: "eyJ...",     // the anon public key, NOT service_role
```

Test locally with `cd web && python3 -m http.server 8000` and run a real comparison. The first run takes an extra 1–2 minutes while the GPU loads the models.

### 5. Put the site online (Vercel)

1. This code lives at [github.com/sijiawen/motiontracking](https://github.com/sijiawen/motiontracking).
2. At [vercel.com](https://vercel.com), click **Add New → Project**, import the repo, and set **Root Directory** to `web`. Framework preset: **Other**. Then click **Deploy**.
3. From then on, every change pushed to the `main` branch redeploys the site automatically. Backend changes (anything in `backend/`) still need `python3 -m modal deploy backend/modal_app.py`.
3. Lock the API to your site. Re-run the `modal secret create` command with `--force` and `ALLOWED_ORIGINS=https://your-site.vercel.app`, then run `modal deploy backend/modal_app.py` again.

---

## Seeing what everyone is checking

- **Admin page:** go to `your-site/admin.html` and enter your `ADMIN_KEY`. You get every comparison, newest first, with both videos playable inline, the score and per-model scores, an anonymous visitor ID, and a CSV download.
- **Supabase:** *Table Editor → comparison_overview* has the same data as a spreadsheet. *Storage → videos* holds the raw uploads, in one folder per comparison.

## Testing on your own AI video set (calibration)

Until you do this, results show an **"Uncalibrated preview score"** badge. Raw scores aren't meaningful percentages yet; for example, two unrelated people standing still can look 50% "similar".

1. Make a folder, e.g. `my_test_set/`, with your videos and a `pairs.csv`:

   ```csv
   video_a,video_b,label,note
   maya_original.mp4,maya_faceswap_ad.mp4,match,face + script swapped
   maya_original.mp4,maya_flipped_ad.mp4,match,mirrored and cropped
   maya_original.mp4,maya_other_video.mp4,unrelated,same creator different footage
   maya_original.mp4,stock_dancer.mp4,unrelated,
   ```

   - `match` means B was made from A's footage. `unrelated` means it wasn't.
   - Include **same person, different video** pairs labelled `unrelated`, especially ones filmed in the **same room** or using the **same script template**. Because the final score takes the highest test, these are the pairs most likely to cause false alarms.
   - Aim for at least 10 of each label; 30+ is better.
2. Run `python backend/calibrate.py my_test_set`. It scores every pair on the GPU, in parallel.
3. Open `my_test_set/calibration_report.html` to see:
   - how well each model separates copies from unrelated videos (AUC: 1.0 is perfect, 0.5 is a coin flip)
   - the score distributions
   - every pair, with misclassified ones in red
4. The script writes `backend/calibration/calibration.json`, which holds the learned per-model weights, the 0%/100% anchors and the verdict thresholds. Run `modal deploy backend/modal_app.py` to make the site use them.

To tweak the analysis without re-running the GPU, change `similarity.py` and run `python backend/calibrate.py my_test_set --reuse`.

## Limits and costs

- **Uploads:** 60 seconds and 50 MB per video. 50 MB is the Supabase free-plan maximum. To raise it, upgrade Supabase to Pro and change `MAX_SECONDS`/`MAX_BYTES` in `modal_app.py`, the bucket limit in `schema.sql`, and `web/config.js`.
- **Daily limit:** currently unlimited (`DAILY_LIMIT = 0`). To cap it, set a number in `modal_app.py` and `web/config.js`. Your own browser bypasses the limit if it sends the admin key.
- **GPU cost:** roughly 3–5 minutes of H100 time per comparison (including GPU start-up). A 60 s + 60 s pair is 240 frames × 5 models. At Modal's ~$0.0011/second that's about $0.20–0.35 per comparison. At most 2 GPUs run at once (`max_containers`), and they shut off 2 minutes after the last job.
- **Free-plan storage:** 1 GB on Supabase. That fills up after roughly 20–40 comparisons of full-size videos, so plan to upgrade or periodically clear old videos.

## How the score works

1. ffmpeg samples 2 frames per second from each video.
2. All five models run on every frame. The **person detector + pose model** follow the largest person in frame. **Face landmarks are excluded** from the pose score because face swaps change them.
3. Each frame becomes a compact fingerprint, cropped to the person so position and zoom don't matter.
4. Every frame of A is compared with every frame of B, separately for each model.
5. **Dynamic time warping** finds the best alignment. This handles trimmed, sped-up, slowed-down copies and dropped frames.
6. Everything is repeated with B **mirrored**, and the better orientation is kept.
7. The final score is a weighted average along the matched path, rescaled by your calibration.

Known limitations of this first version:

- It works best with one main person on screen.
- It compares bodies, not voices or backgrounds.
- A copy that re-performs the same choreography with a different body could score moderately.

Calibration on real examples is how you'll learn where these limits actually bite.

## Before you launch publicly

- **Licence:** the Sapiens2 licence prohibits use "for biometric processing" and for surveillance, and requires compliance with biometric privacy laws (BIPA, CCPA, GDPR). This tool is designed to compare footage, not identify people. Have a lawyer confirm that framing before charging for it or scaling up.
- **Privacy notice:** update the contact email in `web/index.html` (search for `hello@example.com`) and have the privacy text reviewed.
- Uploads may contain other people's likenesses. Decide how long you keep videos, and add a scheduled cleanup if needed.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Upload failed (4xx)" | Check `SUPABASE_URL` and the anon key in `config.js`, and that `schema.sql` ran (buckets exist). |
| Browser console shows a CORS error | `ALLOWED_ORIGINS` in the Modal secret must match your site's address exactly, or be `*`. |
| Stuck on "Starting GPU" | First run after idle loads ~30 GB of models (1–2 min). Check *Modal dashboard → Apps → origin-video-compare → Logs*. |
| `download_models` says 401/403 | The models are gated: do step 2 and add `HF_TOKEN` to the secret. |
| "No person was detected" | One video has no clearly visible person for most of its length. |
| Error mentioning a missing config file | The Sapiens2 repo changed. Pin it by editing the `git clone` line in `modal_app.py` to a known commit. |

Run the scoring math tests locally with `pip install pytest && python -m pytest backend -q`.
