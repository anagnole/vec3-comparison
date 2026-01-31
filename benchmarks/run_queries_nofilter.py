import random
import numpy as np
from benchmarks.run_queries_all import run_one_mode

def main():
    random.seed(42)
    np.random.seed(42)
    run_one_mode(mode_tag="nofilter", with_filters=False)

if __name__ == "__main__":
    main()
