import os
from cox.store import Store
import shutil
import subprocess
from cox.readers import CollectionReader

## Code sample to go alongside Walkthrough #2 in README.md

OUT_DIR = '/tmp/cox_example/'

try:
    shutil.rmtree(OUT_DIR)
except:
    pass

os.mkdir(OUT_DIR)

if __name__ == "__main__":
    for slope in range(5):
        store = Store(OUT_DIR)
        store.add_table('metadata', {'slope': int})
        store.add_table('line_graphs', {'mx': int, 'mx^2': int})
        store['metadata'].append_row({'slope': slope})

        for x in range(100):
            store.log_table_and_tb('line_graphs', {
                'mx': slope*x, 
                'mx^2': slope*(x**2)
            })
            store['line_graphs'].flush_row()

        store.close()

    ### Collection reading
    print("Done experiments, printing results...")
    reader = CollectionReader(OUT_DIR)
    print(reader.df('line_graphs'))

    print("Starting tensorboard:")
    subprocess.run(["python", "-m", "cox.tensorboard_view", "--logdir",
        OUT_DIR, "--format-str", "slope-{slope}", "--filter-param", "slope",
        "[1-3]", "--metadata-table", "metadata"])