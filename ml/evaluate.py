import sys
import json
import os

THRESHOLD = 0.80 

def main():
    data = json.load(sys.stdin)
    if data["accuracy"] >= THRESHOLD:
        print("Gate Passed!")
        sys.exit(0)
    else:
        print("Gate Failed: Accuracy too low.")
        sys.exit(1)

if __name__ == "__main__":
    main()