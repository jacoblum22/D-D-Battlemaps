#!/usr/bin/env python3
"""
Check caption file counts
"""


def main():
    with open("train_captions.txt", "r") as f:
        train_captions = [line.strip() for line in f if line.strip()]
    with open("val_captions.txt", "r") as f:
        val_captions = [line.strip() for line in f if line.strip()]
    with open("test_captions.txt", "r") as f:
        test_captions = [line.strip() for line in f if line.strip()]

    print(
        f"Caption files: train={len(train_captions)}, val={len(val_captions)}, test={len(test_captions)}, total={len(train_captions)+len(val_captions)+len(test_captions)}"
    )


if __name__ == "__main__":
    main()
