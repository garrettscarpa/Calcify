# Calcify

A flexible, GUI-based application for quantifying events from calcium imaging data. Calcify works with plain **CSV** files or with **Suite2p** (`.npy`) output folders, and runs as a standalone desktop app — no Python installation or coding required.

---

## Download & Install

Grab the latest build for your machine from the [**Releases**](../../releases) page.

| Platform | Download |
| --- | --- |
| macOS (Apple Silicon — M1/M2/M3/M4) | `Calcify-macOS-AppleSilicon.zip` |
| macOS (Intel) | `Calcify-macOS-Intel.zip` |
| Windows | `Calcify-v1.0.2-windows.zip` |

Unzip the download, then move `Calcify.app` (macOS) into your **Applications** folder, or run it from wherever you like.

### First launch on macOS

Because the app isn't notarized through the Apple Developer Program, macOS Gatekeeper will block it on the first open. To get around this:

- **Right-click** (or Control-click) `Calcify.app` → choose **Open** → confirm **Open** in the dialog. You only need to do this once.
- If it still refuses, open **Terminal** and run:
  ```bash
  xattr -dr com.apple.quarantine /path/to/Calcify.app
  ```

After the first successful launch, you can open it normally.

---

## How to Use

### 1. Launch Calcify and point it at your data

Open the app to bring up the main window. Here you select the folder containing your files and choose options such as the data format and whether the data need to be transposed.

- **Transpose:** If the ROI labels are **column** titles, the data **need** to be transposed. If the ROI labels are **row** titles, they do **not**.
- When you select a folder, Calcify automatically detects and lists every `.npy` or `.csv` file inside. Click the one you want to process to highlight it.

> ⚠️ **Suite2p users:** You **must** keep the option to **remove non-cell ROIs** enabled (it is checked by default). Disabling it will include ROIs that Suite2p flagged as non-cells.

<img width="884" alt="Calcify main window" src="https://github.com/user-attachments/assets/3cb6df7e-acbc-4ac1-9209-011f3b19a5d5" />

### 2. Choose file type, select the parent folder, and set parameters

Select the file type, then select the **parent folder** holding the file(s) you want to analyze — **not** the file itself. As long as the correct file type is selected, the file appears in the top window; click it to highlight it.

Enter the parameters for your project, including the **sampling frequency (fs)**. Transpose the data if needed (Suite2p data typically does **not** need transposing; CSV data often **does**).

- **Treatment / stimulus:** If your recording has one, define a **pre** and **post** period. Detected peaks are then automatically assigned to the appropriate group.
- **Drift:** If there's drift at the beginning of the recording, you can truncate it.

### 3. Plot all ROIs and check for artifacts

Click **Plot all ROIs** to check for artifacts that appear across every ROI, and to confirm the **Transpose data?** toggle is set correctly.

If motion artifacts are present, you can:

- Try **Auto Remove Motion Artifacts** (may need some tuning for your data), or
- Highlight a portion of the signal and click **Remove Selected Region**. For each ROI, Calcify interpolates the signal between the start and end of the highlighted region.

The example below does **not** contain motion artifacts:

<img width="1017" alt="Plot all ROIs — clean signal" src="https://github.com/user-attachments/assets/876506ac-0cc3-4e91-873e-e391d290cb9d" />

And here is how artifact removal behaves when there **are** substantial motion artifacts:

<img width="561" alt="Motion artifact example 1" src="https://github.com/user-attachments/assets/fa38e229-a6bc-4aa4-a9a9-cf7158451e6b" />
<img width="561" alt="Motion artifact example 2" src="https://github.com/user-attachments/assets/74521028-bb04-4aab-b9e4-7a30befb5665" />
<img width="1125" alt="Motion artifact example 3" src="https://github.com/user-attachments/assets/e3a3df8b-88c2-40c0-8458-bf265d78b92c" />

### 4. Run the analysis

Click **Run Analysis** to open the interactive peak-review window.

<img width="1193" alt="Run Analysis — interactive peak review" src="https://github.com/user-attachments/assets/4634f452-2af1-413c-b03d-bf7deb0952fb" />

Controls:

- **Left / Right arrows** — cycle between peaks
- **Next / Last** — cycle between ROIs
- **Reject** — remove a peak
- **Add Peak** — toggles add-peak mode; while active, click anywhere in the top window to create a new peak at that location
- **Export** — saves all your changes so they persist when you reload the file, and writes all ROI statistics to the same location as your data file

Remember to click **Export** to save your work.

---

## Sample Data

Sample CSV and Suite2p datasets are available as a separate download on the [Releases](../../releases) page (kept out of the main repository to keep it lightweight). Download and unzip them to try Calcify without your own recordings.

---

## Known Issues

- **The Run Analysis window can freeze sporadically.** Until this is resolved, **save frequently** by clicking **Export** — for example, every few ROIs. If this is seriously disrupting your work, please open an issue and I'll prioritize the fix.

---

## Running from Source (for developers)

If you'd rather run or modify the code directly:

```bash
git clone https://github.com/garrettscarpa/Calcify.git
cd Calcify
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

To build a standalone app yourself, use the included build scripts (`build_mac.sh` on macOS, `build_windows.bat` on Windows), which wrap PyInstaller and the provided `Calcify.spec`.

---

## Feedback

Found a bug or have a feature request? Please [open an issue](../../issues).
