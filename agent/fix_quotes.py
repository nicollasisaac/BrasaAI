import sys, re, pathlib

MAP = {
    "\u201C": '"', "\u201D": '"',    # double smart
    "\u2018": "'", "\u2019": "'",    # single smart
    "\u00A0": " ",                   # nbsp
}

def sanitize_text(s: str) -> str:
    return "".join(MAP.get(ch, ch) for ch in s)

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_quotes.py <file_or_dir>")
        return
    p = pathlib.Path(sys.argv[1])
    if p.is_file():
        txt = p.read_text(encoding="utf-8", errors="ignore")
        p.write_text(sanitize_text(txt), encoding="utf-8")
        print(f"Sanitized: {p}")
    else:
        for f in p.rglob("*.py"):
            txt = f.read_text(encoding="utf-8", errors="ignore")
            f.write_text(sanitize_text(txt), encoding="utf-8")
            print(f"Sanitized: {f}")

if __name__ == "__main__":
    main()
