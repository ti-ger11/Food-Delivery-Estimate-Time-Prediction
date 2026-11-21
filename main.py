"""
Runs the full pipeline in order.
    python main.py            -> Phase 1 + Phase 2
    python main.py --phase 1  -> only Phase 1
    python main.py --phase 2  -> only Phase 2
"""
import argparse
import time

import run_phase1
import run_phase2

parser = argparse.ArgumentParser()
parser.add_argument("--phase", type=int, choices=[1, 2], help="run only this phase")
args = parser.parse_args()

start = time.time()

if args.phase in (None, 1):
    print("=" * 50, "\nPHASE 1: Load -> Clean -> Features\n" + "=" * 50)
    run_phase1.main()

if args.phase in (None, 2):
    print("=" * 50, "\nPHASE 2: Train -> Evaluate -> Save best model\n" + "=" * 50)
    run_phase2.main()

print(f"\nAll done in {(time.time() - start) / 60:.1f} minutes")
print("Now start the app:  streamlit run app.py")