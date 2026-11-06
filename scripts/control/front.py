import matplotlib.font_manager as fm

for f in fm.findSystemFonts():
    if "Times" in f or "times" in f:
        print(f)