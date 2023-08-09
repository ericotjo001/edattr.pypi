import argparse
import numpy as np
import matplotlib.pyplot as plt
from edattr.endorse import edattr_to_feature_attr

def main(**kwargs):
    endorsement = {0:2, 2:1, 3:1}
    max_endo = np.max([endo for _,endo in endorsement.items()])

    n_features = 7
    edattr = edattr_to_feature_attr(endorsement, n_features)
    x_labels = [f'x{i}' for i in range(n_features)]

    plt.figure()
    plt.gcf().add_subplot(111)
    plt.gca().barh(range(len(edattr)),edattr[::-1],tick_label=x_labels[::-1])
    plt.gca().set_xlim(None,max_endo+0.5)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=None)        
    args, unknown = parser.parse_known_args()
    kwargs = vars(args)  # is a dictionary        

    main(**kwargs)