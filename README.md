
# 🔧 OCR & AI Recognition Studio

A world-class, **100% local**, browser-based application for:

- 📄 **OCR**: Convert scanned PDFs and images into searchable text using **Tesseract.js 5** (13 languages).
- 🧠 **AI Recognition**: Train a custom model to identify **logos**, **signatures**, and **stamps** using **MobileNet v2 + KNN transfer learning** (TensorFlow.js).
- 💾 **Compressed Database**: Training samples stored in IndexedDB with **JPEG compression (~70% smaller)**.
- 📤 **Export**: Save OCR output as **searchable PDFs** or plain text files.

> **Privacy first**: Nothing ever leaves your machine. All OCR, ML inference, and storage is 100% local.

---

## 📋 Table of Contents

1. [What You Need (Prerequisites)](#-what-you-need-prerequisites)
2. [Step 1: Check What You Already Have](#-step-1-check-what-you-already-have)
3. [Step 2: Install What You're Missing](#-step-2-install-what-youre-missing)
4. [Step 3: Run the App (3 Easy Methods)](#-step-3-run-the-app-3-easy-methods)
5. [First-Time Use Walkthrough](#-first-time-use-walkthrough)
6. [Troubleshooting](#-troubleshooting)
7. [FAQ](#-faq)
8. [How It Works](#-how-it-works)

---

## 📦 What You Need (Prerequisites)

You already have **VS Code** ✅ — great! Here's everything else you'll need. **Don't install anything yet** — first check [Step 1](#-step-1-check-what-you-already-have) to see what's already on your system.

| # | Software | Purpose | Required? | Download Link |
|---|---|---|---|---|
| 1 | **VS Code** | Code editor (you have it) | ✅ Yes | [code.visualstudio.com](https://code.visualstudio.com) |
| 2 | **A modern browser** | To run the app | ✅ Yes | [Chrome](https://www.google.com/chrome) / [Edge](https://www.microsoft.com/edge) / [Firefox](https://www.mozilla.org/firefox) |
| 3 | **Node.js (LTS)** | Runs a local web server | ⭐ Recommended | [nodejs.org](https://nodejs.org) |
| 4 | **VS Code "Live Server" extension** | Easiest one-click option | Alternative to Node | Install inside VS Code |
| 5 | **Python 3** | Alternative local server | Optional fallback | [python.org](https://www.python.org/downloads) |

> ℹ️ **You only need ONE of #3, #4, or #5** — not all three. The Live Server extension is the easiest if you're not comfortable with a terminal.

---

## 🔍 Step 1: Check What You Already Have

Let's check what's already installed on your computer **before** downloading anything.

### ✅ Check 1 — VS Code Version

1. Open **VS Code**.
2. Click **Help** → **About** (on macOS: **Code** → **About Visual Studio Code**).
3. Make sure version is **1.80+** (the current version as of 2024 is 1.90+).

If older, update via **Help → Check for Updates**.

---

### ✅ Check 2 — Your Browser

Open **Chrome**, **Edge**, or **Firefox** and paste into the address bar:

| Browser | Paste this URL |
|---|---|
| Chrome | `chrome://version` |
| Edge | `edge://version` |
| Firefox | `about:support` |

**Minimum versions required:**
- Chrome/Edge: **100 or newer**
- Firefox: **100 or newer**
- Safari: **15 or newer**

All modern browsers (Chrome 120+, Edge 120+) released since 2022 will work perfectly.

---

### ✅ Check 3 — Node.js (Recommended)

Open a **terminal** and run the check:

#### 🪟 Windows
1. Press `Windows` key + `R`, type `powershell`, press **Enter**.
2. Or: Press `Windows` key, type **PowerShell**, press **Enter**.
3. In the blue window, type:
   ```powershell
   node --version
   npm --version
   ```
4. Press Enter.

#### 🍏 macOS
1. Press `Cmd` + `Space`, type **Terminal**, press **Enter**.
2. Type:
   ```bash
   node --version
   npm --version
   ```

#### 🐧 Linux
Open Terminal (usually `Ctrl` + `Alt` + `T`) and run:
```bash
node --version
npm --version
```

**Expected output:**
```
v20.11.0      ← Node version (any v18+ is fine)
10.2.4        ← npm version
```

**If you get:**
- ✅ Version numbers → **You're done! Skip to [Step 3](#-step-3-run-the-app-3-easy-methods).**
- ❌ `'node' is not recognized...` or `command not found: node` → **Install Node.js in [Step 2](#-step-2-install-what-youre-missing).**

> 💡 **Pro tip**: VS Code has a built-in terminal. Press `` Ctrl + ` `` (backtick, the key left of `1`) to open it. On Mac: `` Cmd + ` ``. You can run all terminal commands from there.

---

### ✅ Check 4 — Python (Optional Fallback)

Only needed if you don't want Node.js or the Live Server extension.

#### 🪟 Windows
```powershell
python --version
```

#### 🍏 macOS / 🐧 Linux
```bash
python3 --version
```

**Expected output:**
```
Python 3.11.6    ← any 3.7+ works
```

---

## ⬇️ Step 2: Install What You're Missing

### 🟢 Installing Node.js (Recommended — 5 minutes)

This is the **best option** because it's fast, tiny, and reliable.

#### 🪟 Windows Installation

1. Visit 👉 **[https://nodejs.org](https://nodejs.org)**
2. Click the big green **"LTS"** button (left side — says "Recommended For Most Users").
3. A file named `node-v20.x.x-x64.msi` downloads.
4. **Double-click the installer**.
5. Click **Next** through every screen:
   - ✅ Accept license
   - ✅ Use default install location (`C:\Program Files\nodejs\`)
   - ✅ Default features
   - ✅ **"Automatically install necessary tools"** — leave **unchecked** (we don't need it)
6. Click **Install** → wait ~1 minute → **Finish**.
7. **🔴 IMPORTANT: Close ALL PowerShell/Terminal windows** and reopen a new one.
8. Verify:
   ```powershell
   node --version
   ```
   You should see something like `v20.11.0`. 🎉

#### 🍏 macOS Installation

**Option 1 — installer (easiest):**
1. Go to [nodejs.org](https://nodejs.org) → click the **LTS** button.
2. Download the `.pkg` file (e.g., `node-v20.x.x.pkg`).
3. Double-click → follow the installer.
4. Open a new Terminal → `node --version`.

**Option 2 — Homebrew (if you have it):**
```bash
brew install node
```

#### 🐧 Linux Installation (Ubuntu/Debian)

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
node --version
```

---

### 🌐 Alternative: Install "Live Server" VS Code Extension (No Terminal Needed!)

If you don't want to install Node.js, use this VS Code extension instead. It's the simplest option.

1. Open **VS Code**.
2. On the left sidebar, click the **Extensions** icon (4 squares, or press `Ctrl + Shift + X` / `Cmd + Shift + X`).
3. In the search box at the top, type: **`Live Server`**
4. Find the one by **"Ritwick Dey"** (it has ~47 million downloads — a blue icon).
5. Click **Install**.
6. Done! See [Step 3 Method B](#-method-b-vs-code-live-server-no-terminal) to use it.

---

### 🐍 Installing Python (Optional — Skip if You Have Node)

Only install this if both Node.js and Live Server failed.

#### Windows
1. Visit [python.org/downloads](https://www.python.org/downloads).
2. Click **Download Python 3.x.x**.
3. **🚨 CRITICAL**: On the first installer screen, **check the box** that says **"Add python.exe to PATH"** at the bottom.
4. Click **Install Now** → **Close**.
5. Open a new PowerShell → `python --version`.

#### macOS
Python 3 is usually already installed. If not:
```bash
brew install python
```

#### Linux
```bash
sudo apt install python3
```

---

## 🚀 Step 3: Run the App (3 Easy Methods)

Pick **ONE** method based on what you installed. All three produce the same result.

### 📁 First: Get the Project Onto Your Computer

1. If you have a `.zip` file:
   - Right-click → **Extract All** (Windows) / double-click (macOS).
   - Choose a memorable folder, e.g. `C:\Projects\ocr-ai-studio` or `~/Documents/ocr-ai-studio`.
2. If you have Git:
   ```bash
   git clone <your-repo-url> ocr-ai-studio
   ```

### 📂 Open the Project in VS Code

1. Open **VS Code**.
2. **File → Open Folder…**
3. Select the `ocr-ai-studio` folder you just extracted.
4. If VS Code asks **"Do you trust the authors?"** → click **Yes, I trust the authors**.

You should now see all the project files (`index.html`, `src/`, `README.md`, etc.) in the left Explorer panel.

---

### 🟢 Method A: Node.js (Recommended)

1. In VS Code, open a terminal: **Terminal → New Terminal** (or press `` Ctrl + ` ``).
2. In the terminal (which is already inside your project folder), type:
   ```bash
   npx serve .
   ```
3. The first time, it will ask **"Ok to proceed? (y)"** — press **`y`** + Enter.
4. You'll see:
   ```
   ┌──────────────────────────────────────────┐
   │   Serving!                               │
   │   - Local:    http://localhost:3000      │
   └──────────────────────────────────────────┘
   ```
5. **Hold `Ctrl`** (Windows/Linux) or **`Cmd`** (macOS) and **click** the URL. Or copy-paste it into your browser.
6. 🎉 The app loads! First load takes ~15 seconds to fetch ML models.

**To stop**: Click in the terminal, press `Ctrl + C`, confirm with `y`.

---

### 💠 Method B: VS Code Live Server (No Terminal!)

If you installed the Live Server extension instead of Node.js:

1. In VS Code's Explorer (left panel), find the file **`index.html`** at the project root.
2. **Right-click** it → select **"Open with Live Server"**.
3. Or: click the **"Go Live"** button in the bottom-right status bar of VS Code.
4. Your default browser opens automatically at `http://127.0.0.1:5500`.
5. 🎉 Done!

**To stop**: Click the **"Port : 5500"** indicator in the bottom-right of VS Code.

---

### 🐍 Method C: Python (Fallback)

1. Open VS Code terminal (`` Ctrl + ` ``).
2. Type:
   ```bash
   # Windows
   python -m http.server 8080

   # macOS / Linux
   python3 -m http.server 8080
   ```
3. Open browser → **http://localhost:8080**.
4. 🎉 Loaded!

**To stop**: Press `Ctrl + C` in the terminal.

---

## 🎬 First-Time Use Walkthrough

Once the app opens in your browser:

### 1️⃣ Dashboard
You'll land on the **Dashboard** with live stats of your database.

### 2️⃣ Try OCR (Document Scanning)
1. Click **OCR Workspace** in the left sidebar.
2. **Upload** a scanned PDF or image (drag-and-drop works).
3. Select a **language** (default: English).
4. Click **Extract Text** — the first OCR run downloads the language model (~10 MB, one-time).
5. Edit the extracted text on the right if needed.
6. Click **Save Searchable PDF** or **Save Text File**.

### 3️⃣ Train AI Recognition
1. Click **AI Recognition** in the sidebar.
2. Choose a category: **Logo**, **Signature**, or **Stamp**.
3. Enter a **label** (e.g. "Acme Corporation").
4. Drop **3–10 sample images** of that item.
5. Click **Save & Train** — samples are compressed and indexed instantly.
6. Repeat for other labels/categories.

### 4️⃣ Identify an Image
1. Switch to **🔎 Predict** mode (top of the page).
2. Upload a new image.
3. Click **Identify** — see top-5 matches with confidence percentages.

### 5️⃣ Manage Your Data
Click **Database** in the sidebar to:
- Browse all training samples (filter by category).
- View all historic OCR documents.
- Delete individual items or clear everything.
- See compression savings in real-time.

---

## 🔧 Troubleshooting

<details>
<summary><b>🚫 "'node' is not recognized" or "command not found: node"</b></summary>

Node.js isn't installed, or the installer didn't add it to your PATH.
- Close **ALL** terminal/PowerShell windows and open a **fresh** one (the old one doesn't know about Node yet).
- On Windows, reboot if the above doesn't work.
- Re-install Node.js from [nodejs.org](https://nodejs.org) — make sure you use the **LTS** version.
</details>

<details>
<summary><b>🚫 "'python' is not recognized" (Windows)</b></summary>

You forgot to tick **"Add python.exe to PATH"** during install.
- Uninstall Python from **Settings → Apps**.
- Re-install from [python.org](https://www.python.org/downloads) and **check the PATH box** this time.
</details>

<details>
<summary><b>🚫 "Port 3000 is already in use"</b></summary>

Another app is using that port. Use a different one:
```bash
npx serve . -l 4000
# or
python -m http.server 9090
```
Then open `http://localhost:4000` instead.
</details>

<details>
<summary><b>🚫 VS Code Live Server button doesn't appear</b></summary>

- Make sure you installed the extension by **Ritwick Dey** (not a different one).
- Reload VS Code: `Ctrl + Shift + P` → type **"Reload Window"** → Enter.
- Make sure you opened the **folder** (not just the file) using **File → Open Folder**.
</details>

<details>
<summary><b>🚫 Blank white page / app doesn't load</b></summary>

- Open browser DevTools (press `F12`) → **Console** tab → look for red errors.
- **Most common cause**: you opened `index.html` by **double-clicking it** (URL starts with `file://`). That doesn't work — you **must** use a local server (Methods A, B, or C above).
- Check your browser version (Chrome/Edge 100+ required).
</details>

<details>
<summary><b>🐢 OCR is very slow</b></summary>

- First OCR run downloads a language model (~10 MB) — subsequent runs are 5–10× faster.
- Large PDFs (50+ pages) take time; each page is ~2–5 sec.
- Close other heavy browser tabs to free up RAM.
</details>

<details>
<summary><b>❌ "Failed to load MobileNet" / blank AI page</b></summary>

You need an internet connection on **first load** to fetch TensorFlow.js models (~15 MB). They're cached afterward — you can go fully offline.
</details>

<details>
<summary><b>💾 Data disappeared after clearing browser data</b></summary>

The app uses **IndexedDB** which is cleared when you "Clear browsing data → Cookies and site data". To preserve data, avoid this action, or use a desktop wrapper (Tauri/Electron) which persists storage natively.
</details>

<details>
<summary><b>🔒 CORS / "file://" errors in console</b></summary>

You **cannot** open `index.html` by double-clicking it. You **must** serve it over HTTP (that's what Methods A/B/C do). Browser security blocks ES modules on `file://`.
</details>

---

## ❓ FAQ

**Q: Does it work offline?**
A: Yes — after the first load. Models are cached in the browser.

**Q: Where is my data stored?**
A: In your browser's **IndexedDB** under the database name `ocr_ai_studio`. Per-browser, per-origin.

**Q: Can I back up my training data?**
A: Currently via IndexedDB export tools (e.g. Chrome DevTools → Application → IndexedDB → Export). A built-in Import/Export feature is on the roadmap.

**Q: What image formats are supported?**
A: PNG, JPEG, WebP, GIF, BMP for images. PDF (vector + scanned) for documents.

**Q: How many languages does OCR support?**
A: 13 are pre-configured; Tesseract actually supports **100+** — edit `src/features/ocr/OcrWorkspace.tsx` → `LANGS` array to add more.

**Q: Does it use my GPU?**
A: TensorFlow.js auto-uses **WebGL** (GPU) when available, falling back to WASM/CPU otherwise.

**Q: Do I need to install anything inside the project (like `npm install`)?**
A: **No!** All dependencies are loaded from CDN at runtime via ES modules. The project folder contains only source files — nothing to compile.

**Q: Is this commercial-use safe?**
A: Tesseract (Apache 2.0), TF.js (Apache 2.0), and MobileNet (Apache 2.0) are all commercially permissive. Check your own use case.

---

## 🧠 How It Works

### OCR Pipeline
```
PDF/Image → pdf.js renders page → canvas → Tesseract.js 5 → Text + bounding boxes → jsPDF → Searchable PDF
```

### AI Recognition Pipeline
```
Image → MobileNet v2 (frozen) → 1024-dim embedding → KNN Classifier (k=5) → Label + confidence
```

Because MobileNet is pre-trained on ImageNet, you only need **3–10 samples per class** for accurate recognition. No training time, no GPU required.

### Compression Strategy
- Resize → max 480 px (longest edge)
- Re-encode → JPEG quality 0.72
- Store → Base64 in IndexedDB
- Savings → **60–75%** vs original PNG

---

## 📦 Dependencies

All loaded via ES Modules CDN (no `npm install` needed for the web app):

| Library | Version | Purpose |
|---|---|---|
| react | 18.2.0 | UI framework |
| tesseract.js | 5.1.0 | OCR engine |
| pdfjs-dist | 4.0.379 | PDF rendering |
| jspdf | 2.5.1 | PDF generation |
| @tensorflow/tfjs | 4.17.0 | ML runtime |
| @tensorflow-models/mobilenet | 2.1.1 | Feature extractor |
| @tensorflow-models/knn-classifier | 1.2.4 | Transfer learning |

---

## 📜 License

MIT — use freely, commercially or personally.

---

## 🙏 Credits

Built with ❤️ using Tesseract.js, TensorFlow.js, pdf.js, React, and jsPDF.

**Enjoy! 🚀** If this helps you, star the repo.
