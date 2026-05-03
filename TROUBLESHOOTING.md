
# 🩺 Blank Screen? Read This

If you saw a **dark blank screen** instead of the app, here's what was happening and how it's now fixed.

## What Was Wrong

Browsers can't execute `.tsx` (TypeScript) files directly — they only understand plain JavaScript. The original setup tried to load TSX files via `<script type="module">`, which fails silently in some browsers (showing a blank screen with no visible error).

## What's Fixed Now

The project now includes an **in-browser compiler** (Babel Standalone) that:

1. 📥 Fetches all your `.ts` / `.tsx` files
2. 🔄 Compiles them to JavaScript in real-time
3. 🔗 Wires up imports via Blob URLs (native ES modules)
4. ⚡ Runs the React app

**Bonus**: If anything ever fails, you'll now see a **clear error panel** with diagnostics instead of a blank screen.

## How to Launch (All Steps)

### If you're using VS Code Live Server:

1. Make sure the project is fully extracted (no nested zip folders).
2. In VS Code, open the folder: **File → Open Folder**.
3. In the left Explorer, **right-click `index.html`** → **Open with Live Server**.
4. Browser opens at `http://127.0.0.1:5500`.
5. You should now see:
   - An animated logo with "Initializing…"
   - Status changing to "Loading source files…" → "Compiling TypeScript…" → "Starting React app…"
   - The full dashboard appears within ~5–15 seconds on first load.

### If it STILL shows a problem:

The error panel will now tell you exactly what's wrong. Common fixes:

| Error | Fix |
|---|---|
| `Failed to fetch src/index.tsx (HTTP 404)` | You opened the wrong folder. Make sure `index.html` and the `src/` folder are siblings. |
| `Babel Standalone failed to load` | No internet, or firewall blocks `unpkg.com`. |
| `Failed to fetch ... esm.sh` | No internet, or firewall blocks `esm.sh`. Try a different network. |
| `Cannot resolve import "..."` | A file is missing from the project folder. Re-extract the zip. |
| URL in browser starts with `file://` | You double-clicked `index.html`. Use **Live Server** or `npx serve .` instead. |

## How to Verify Internet Connectivity to CDNs

Open these URLs in a new browser tab — each should return JavaScript (not an error):

- https://unpkg.com/@babel/standalone@7.24.0/babel.min.js
- https://esm.sh/react@18.2.0

If either fails, your network is blocking them. Try a phone hotspot to confirm.

## Quick Sanity Check Commands

In VS Code terminal:

```bash
# Make sure you're in the right folder (should list index.html)
ls
# or on Windows
dir
```

You should see:
```
index.html    src/    README.md    ...
```

If `src/` is missing or `index.html` is in a different location, fix that first.

---

After these changes, the app will start reliably on **any local server** (Live Server, `npx serve`, `python -m http.server`) with clear feedback if anything goes wrong.
