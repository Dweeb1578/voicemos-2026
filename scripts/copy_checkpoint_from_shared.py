"""Run this in Colab after mounting Drive to copy epoch_005.pt from shared Drive."""
import glob
import os
import shutil

dst_dir = "/content/drive/MyDrive/voicemos2026/checkpoints"
os.makedirs(dst_dir, exist_ok=True)

# Find shared folder
candidates = (
    glob.glob("/content/drive/Shareddrives/*/voicemos2026/checkpoints/epoch_005.pt")
    + glob.glob("/content/drive/MyDrive/.shortcut-targets-by-id/*/voicemos2026/checkpoints/epoch_005.pt")
)

if not candidates:
    print("Could not find shared checkpoint. Paths searched:")
    print("  /content/drive/Shareddrives/*/voicemos2026/checkpoints/")
    print("  /content/drive/MyDrive/.shortcut-targets-by-id/*/voicemos2026/checkpoints/")
    print("\nList what's available:")
    for p in glob.glob("/content/drive/**", recursive=True):
        if "voicemos" in p.lower():
            print(" ", p)
else:
    src = candidates[0]
    dst = os.path.join(dst_dir, "epoch_005.pt")
    shutil.copy(src, dst)
    print(f"Copied: {src} -> {dst}")
