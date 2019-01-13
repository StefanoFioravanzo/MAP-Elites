import os
import shutil
from pathlib import Path


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f
            

root_log = Path('log/complete_logs/run_10D_standard')
for ld in listdir_nohidden(root_log):
    log_path = root_log / ld

    names = list()
    for d in listdir_nohidden(log_path):
        if d != "plots":
            names.append("{0:02d}".format(int(d)))

    # create directory to store plots
    directory = log_path / "plots"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # iterate the folders and copy the plot to the plots dir
    for d in listdir_nohidden(log_path):
        if d != "plots":
            if (log_path / "plots/{0:02d}.pdf".format(int(d))).is_file():
                os.remove(log_path / "plots/{0:02d}.pdf".format(int(d)))
            shutil.copy(log_path / d / "heatmap.pdf", log_path / "plots/{0:02d}.pdf".format(int(d)))