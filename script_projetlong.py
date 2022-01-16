import os
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Model
import mdtraj as md
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

path = "transfer_2391329_files_66acf08b/TrajR1toR4_Concatenate_PBC_Prot_new.xtc"
path2 = "transfer_2391329_files_66acf08b/TrajR1toR3_Tspo_2mgy_nolig_prod_PBC_Prot.xtc"
topo = "transfer_2391329_files_66acf08b/TrajR1toR3_Tspo_2mgy_nolig_prod_PBC_Protein_average.pdb"
test = "parsedtraj.xtc"


def get_arguments():
    parser = argparse.ArgumentParser(description='Dynamic')
    parser.add_argument('data', nargs=2, metavar='data', type=str, help="Xtc file of molecular dynamic and Pdb file "
                                                                        "of the molecule of interest")
    parser.add_argument('-m', '--mod', nargs=1, required=True,
                        help='type of base classification(mds, pca, tsne, ipca(internal PCA), if you want no base '
                             'classification use raw')
    parser.add_argument('-e', '--epoch', nargs=1, type=int, help="Choose the number of epoch for the learning process"
                                                                 " (default raw = 20 other mods = 50)")
    parser.add_argument('-ca', '--calpha', action='store_true', help="Activated to work only on calpha for raw data "
                                                                     "default=work on all data")
    parser.add_argument('-ag', '--ag', nargs=1,
                        help="group of atom for PCA, MDS. Default is C alpha atoms. Other options for PCA and tsne are"
                             "all(all atoms), backbone(backbone atoms), CA(C alpha atoms), protein"
                             "(protein's atoms)"
                             "options for mds:all(all atoms), backbone(backbone atoms), alpha(C alpha atoms), heavy("
                             "all non hydrogen atoms), minimal(CA, CB, C, N, O atoms")
    parser.add_argument('-i', '--inter', nargs=1, type=int,
                        help="interval to cut xtc file (for computers with not enough "
                             "computation")
    parser.add_argument('-pr', '--perplexity', nargs=1, type=int,
                        help="default = 30, used for tsne, the perplexity is related to \n"
                             "the number of nearest neighbors that is used\n"
                             "in other manifold learning algorithms")
    return parser.parse_args()


def xtcparse(filex, top, inter=1, out="parsedtraj.xtc"):
    """
    Parsing of the trajectory data, can reduce the number of data using the inter argument to keep the conformation
    every interval (inter)
    :param filex: trajectory file
    :param top: topology file(pdb)n
    :param inter: interval
    :param out: name of output file
    :return:name of output file
    """
    traj = md.load(filex, top=top)
    print(f"trajectory before parsing : {traj}")
    traj[::inter].save_xtc(out)
    return out


def run_modetask_PCA(traj, topo, atom_grp="CA", output="pca_result"):
    """
    Runs MODE-TASK PCA
    :param traj: trajectory file
    :param topo: topology file(pdb)
    :param atom_grp: atom group to work on
    :param output: output name
    :return: a folder
    """
    os.system(f"python MODE-TASK/pca.py -t {traj} -p {topo} -out {output} -ag {atom_grp}")


def run_modetask_MDS(traj, topo, atom_grp="CA", output="mds_result"):
    """
    Runs MODE-TASK MDS
    :param traj: trajectory file
    :param topo: topology file(pdb)
    :param atom_grp: atom group to work on
    :param output: output name
    :return: a folder
    """
    os.system(f"python MODE-TASK/mds.py -t {traj} -p {topo} -out {output} -ag {atom_grp}")


def run_modetask_tsne(traj, topo, atom_grp="CA", output="tsne_result", perp=None):
    """
    Runs MODE-TASK t-SNE
    :param traj: trajectory file
    :param topo: topology file(pdb)
    :param atom_grp: atom group to work on
    :param output: output name
    :param perp: value of perplexity
    :return: a folder
    """
    if perp is None:
        os.system(f"python MODE-TASK/tsne.py -t {traj} -p {topo} -out {output} -ag {atom_grp}")
    else:
        os.system(f"python MODE-TASK/tsne.py -t {traj} -p {topo} -out {output} -ag {atom_grp} -pr {perp}")


def run_modetask_intPCA(traj, topo, atom_grp="CA", output="ipca_result"):
    """
    Runs MODE-TASK internal PCA
    :param traj: trajectory file
    :param topo: topology file(pdb)
    :param atom_grp: atom group to work on
    :param output: output name
    :return: a folder
    """
    os.system(f"python MODE-TASK/internal_pca.py -t {traj} -p {topo} -out {output} -ag {atom_grp} -ct angles")


def base_class(arg, traj, topo, argu, atom_grp=None):
    """
    Choosing with which method to work on
    :param arg: method
    :param traj: trajectory file
    :param topo: topology file
    :param argu: recollection of argument
    :param atom_grp: atom group
    :return: a folder
    """
    if arg == "pca":
        print("PCA_launched...")
        if atom_grp is None:
            run_modetask_PCA(traj, topo)
        else:
            run_modetask_PCA(traj, topo, atom_grp=atom_grp)
        print("PCA_done...")
    elif arg == "mds":
        print("MDS_launched...")
        if atom_grp is None:
            run_modetask_MDS(traj, topo)
        else:
            run_modetask_MDS(traj, topo, atom_grp=atom_grp)
        print("MDS_done...")
    elif arg == "tsne":
        print("TSNE_launched...")
        if atom_grp is None:
            if argu.perplexity:
                run_modetask_tsne(traj, topo, perp=argu.perplexity[0])
            else:
                run_modetask_tsne(traj, topo)
        else:
            if argu.perplexity:
                run_modetask_tsne(traj, topo, perp=argu.perplexity[0], atom_grp=atom_grp)
            else:
                run_modetask_tsne(traj, topo, atom_grp=atom_grp)
        print("TSNE_done...")
    elif arg == "ipca":
        if atom_grp is None:
            run_modetask_intPCA(traj, topo)
        else:
            run_modetask_intPCA(traj, topo, atom_grp=atom_grp)
    else:
        print("there is no such argument as mod")
        exit()


def get_dir_file(path):
    """
    Retrieving file names in a list
    :param path: folder in which the files' name must be retrieved
    :return: a list of file names
    """
    files = []
    for file in os.listdir(path):
        files.append(file)
    return files


def get_png_file(path):
    """
    Retrieving graphs path
    :param path: preliminary path of the folder
    :return: list of paths
    """
    png_files = []
    files = get_dir_file(path)
    for file in files:
        if file[-3:] == "png":
            png_files.append(path + "/" + file)
    return png_files


def png_listing(listing):
    """
    Showing graphs from folder
    :param listing: list of graphs/images paths
    :return: show images in a window
    """
    if len(listing) == 3:
        img1 = Image.open(listing[0])
        img2 = Image.open(listing[1])
        img3 = Image.open(listing[2])
        img1.show(title='image1')
        img2.show(title='image2')
        img3.show(title='image3')
    else:
        print("wrong number of data")


def get_arg_file(path):
    """
    retrieve path names of .arg format files
    :param path: preliminary path to folder
    :return: list of paths
    """
    arg_files = []
    files = get_dir_file(path)
    for file in files:
        if file[-3:] == "agr":
            arg_files.append(path + "/" + file)
    return arg_files


def arg_listing(path):
    """
    Retrieving data(coordinates) from .arg file
    :param path: file path
    :return: list of list
    """
    values = []
    with open(path, 'r')as filin:
        for line in filin:
            line = line.strip()
            if not line.startswith("@") and not line.startswith("#") and line != '':
                linesp = line.rstrip().split(" ")
                linesp = list(map(float, linesp))
                values.append(linesp)
    return values


def data_making(values):
    """
    Making training and testing/validation set for pretreated data with MODE-TASK
    :param values: result of either PCA, t-SNE ...
    :return: training dataset, testing/validation dataset
    """
    cut_size = int(len(values) * (8 / 10))
    xtrain = values[0: cut_size]
    xtest = values[cut_size:]
    print("xtrain pre norm")
    print(xtrain[:3])
    min_val = tf.reduce_min(xtrain)
    print(min_val)
    max_val = tf.reduce_max(xtrain)
    print(max_val)
    xtrain = (xtrain - min_val) / (max_val - min_val)
    xtest = (xtest - min_val) / (max_val - min_val)
    print("xtrain post norm")
    print(xtrain[:3])
    print("done printing")
    return np.array(xtrain), np.array(xtest)


def autoencoderMDS(shape):
    """
    autoencoder for MDS pretreated data
    :param shape: shape of data
    :return: autoencoder model
    """
    input_data = tf.keras.Input(shape)
    encoder = layers.Dense(shape, activation='relu')(input_data)
    encoder = layers.Dense(64, activation='relu')(encoder)
    encoder = layers.Dense(48, activation='relu')(encoder)
    encoder = layers.Dense(32, activation='relu')(encoder)
    encoder = layers.Dense(16, activation='relu')(encoder)
    hidden = layers.Dense(8, activation='relu')(encoder)
    decoder = layers.Dense(16, activation='relu')(hidden)
    decoder = layers.Dense(32, activation='relu')(decoder)
    decoder = layers.Dense(48, activation='relu')(decoder)
    decoder = layers.Dense(64, activation='relu')(decoder)
    decoder = layers.Dense(shape, activation='linear')(decoder)
    autoencoded = Model(input_data, decoder)
    return autoencoded


def autoencodertsne(shape):
    """
    autoencoder for t-SNE pretreated data
    :param shape: shape of data
    :return: autoencoder model
    """
    input_data = tf.keras.Input(shape)
    encoder = layers.Dense(shape, activation='relu')(input_data)
    encoder = layers.Dense(64, activation='relu')(encoder)
    encoder = layers.Dense(32, activation='relu')(encoder)
    encoder = layers.Dense(16, activation='relu')(encoder)
    encoder = layers.Dense(8, activation='relu')(encoder)
    hidden = layers.Dense(4, activation='relu')(encoder)
    decoder = layers.Dense(8, activation='relu')(hidden)
    decoder = layers.Dense(16, activation='relu')(decoder)
    decoder = layers.Dense(32, activation='relu')(decoder)
    decoder = layers.Dense(64, activation='relu')(decoder)
    decoder = layers.Dense(shape, activation='linear')(decoder)
    autoencoded = Model(input_data, decoder)
    return autoencoded


def autoencoderPCA(shape):
    """
    autoencoder for PCA pretreated data
    :param shape: shape of data
    :return: autoencoder model
    """
    input_data = tf.keras.Input(shape)
    encoder = layers.Dense(shape, activation='relu')(input_data)
    encoder = layers.Dense(64, activation='relu')(encoder)
    encoder = layers.Dense(48, activation='relu')(encoder)
    encoder = layers.Dense(32, activation='relu')(encoder)
    encoder = layers.Dense(16, activation='relu')(encoder)
    hidden = layers.Dense(8, activation='relu')(encoder)
    decoder = layers.Dense(16, activation='relu')(hidden)
    decoder = layers.Dense(32, activation='relu')(decoder)
    decoder = layers.Dense(48, activation='relu')(decoder)
    decoder = layers.Dense(64, activation='relu')(decoder)
    decoder = layers.Dense(shape, activation='linear')(decoder)
    autoencoded = Model(input_data, decoder)
    return autoencoded


def autoencoderRAW(shape, shapeb):
    """
    autoencoder for raw data
    :param shape: first shape of data
    :param shapeb: second shape of data
    :return: autoencoder model
    """
    input_data = tf.keras.Input(shape=(shape, shapeb,))
    encoder = layers.Dense(shapeb, activation='relu', input_shape=(None, shapeb))(input_data)
    encoder = layers.Dense(150, activation='relu')(encoder)
    encoder = layers.Dense(120, activation='relu')(encoder)
    encoder = layers.Dense(80, activation='relu')(encoder)
    encoder = layers.Dense(64, activation='relu')(encoder)
    hidden = layers.Dense(32, activation='relu')(encoder)
    decoder = layers.Dense(64, activation='relu')(hidden)
    decoder = layers.Dense(80, activation='relu')(decoder)
    decoder = layers.Dense(120, activation='relu')(decoder)
    decoder = layers.Dense(150, activation='relu')(decoder)
    decoder = layers.Dense(shapeb, activation='linear')(decoder)
    autoencoded = Model(input_data, decoder)
    return autoencoded


def autoencoderiPCA(shape):
    """
    autoencoder for PCA pretreated data
    :param shape: shape of data
    :return: autoencoder model
    """
    input_data = tf.keras.Input(shape)
    encoder = layers.Dense(shape, activation='relu')(input_data)
    encoder = layers.Dense(64, activation='relu')(encoder)
    encoder = layers.Dense(48, activation='relu')(encoder)
    encoder = layers.Dense(32, activation='relu')(encoder)
    encoder = layers.Dense(16, activation='relu')(encoder)
    hidden = layers.Dense(8, activation='relu')(encoder)
    decoder = layers.Dense(16, activation='relu')(hidden)
    decoder = layers.Dense(32, activation='relu')(decoder)
    decoder = layers.Dense(48, activation='relu')(decoder)
    decoder = layers.Dense(64, activation='relu')(decoder)
    decoder = layers.Dense(shape, activation='linear')(decoder)
    autoencoded = Model(input_data, decoder)
    return autoencoded


def autoencoder(shape, mod, shapeb=None):
    """
    Autoencoder selection based on method used
    :param shape: main shape for autoencoder
    :param mod: method to use
    :param shapeb: secondary for raw data
    :return: autoencoder model
    """
    if mod == "mds":
        return autoencoderMDS(shape)
    if mod == "tsne":
        return autoencodertsne(shape)
    if mod == "pca":
        return autoencoderPCA(shape)
    if mod == "raw":
        return autoencoderRAW(shape, shapeb)
    if mod == "ipca":
        return autoencoderiPCA(shape)
    else:
        print("no such type")
        exit()


def raw_making(file, topo, ca=0, kdata=None):
    """
    Normalization of raw data
    :param file: trajectory file
    :param topo: topology file(pdb)
    :param ca: if 1 working only on calpha
    :param kdata: list of position for calpha
    :return: normalized data
    """
    traj = md.load(file, top=topo)
    raw_data = traj.xyz
    if ca == 0:
        norm = np.linalg.norm(raw_data)
        norm_raw_data = raw_data / norm
    else:
        new_set = []
        for conf in raw_data:
            inter_set = []
            for i in kdata:
                inter_set.append(conf[i])
            new_set.append(inter_set)
        fraw = np.array(new_set)
        norm = np.linalg.norm(fraw)
        norm_raw_data = fraw / norm
    return norm_raw_data


def raw_set_making(data):
    """
    Making training and testing/validation datasets from raw normalized data
    :param data: data to be split
    :return: training set, testing/validation set
    """
    print(data.size)
    data_split = np.array_split(data, 10)
    xtrain = np.concatenate((data_split[0:8]))
    xtest = np.concatenate(data_split[8:10])
    return np.array(xtrain), np.array(xtest)


def CA_pos(pdbfile):
    """
    Retrieving positions of calphas from pdb file
    :param pdbfile: pdb file
    :return: list of calpha positions and size of it
    """
    data = []
    pos = 0
    with open(pdbfile, 'r')as filin:
        for line in filin:
            if line.startswith("ATOM"):
                rdata = line.split()
                if rdata[2] == 'CA':
                    data.append(pos)
                pos += 1
    return data, len(data)


def main():
    print("start")
    args = get_arguments()
    print("parsing file...")
    if args.inter:
        new_file = xtcparse(args.data[0], args.data[1], inter=args.inter[0])
    else:
        new_file = xtcparse(args.data[0], args.data[1])
    print("parsing done")
    if args.mod[0] == "raw":
        print("parsing raw data :")
        if args.calpha:
            calist, sec_shape = CA_pos(topo)
            raw_data = raw_making(new_file, topo, 1, calist)
        else:
            raw_data = raw_making(new_file, topo)
            sec_shape = len(raw_data[0])
        print("raw data parsed")
        print("Making training and testing set :")
        xtrainr, xtestr = raw_set_making(raw_data)
        print("sets made")
        print("Autoencoder compiling :")
        autoencoderr = autoencoder(sec_shape, "raw", 3)
        autoencoderr.compile(optimizer='adam', loss='mae')
        print("autoencoder compiling done")
        print(autoencoderr.summary())
        print("autoencoder fitting :")
        if args.epoch:
            historyr = autoencoderr.fit(xtrainr, xtrainr, epochs=args.epoch[0], shuffle=True, validation_data=(xtestr,
                                                                                                               xtestr))
        else:
            historyr = autoencoderr.fit(xtrainr, xtrainr, epochs=20, shuffle=True, validation_data=(xtestr, xtestr))
        print("done\nplotting...")
        plt.plot(historyr.history['loss'], label='train')
        plt.plot(historyr.history['val_loss'], label='test')
        plt.legend()
        plt.title('raw/autoencoder loss')
        plt.show()
        print("do you wish to save the encoder?(y/n)")
        choice = input()
        if choice == "y":
            auto = Model(autoencoderr.input, autoencoderr.layers[6].output)
            auto.save("raw_encoder.h5")
            print("model saved")
        print("end")
    else:
        print(f"Base classification with {args.mod[0]} ongoing...")
        if args.ag:
            base_class(f"{args.mod[0]}", new_file, topo, args, atom_grp=args.ag[0])
        else:
            base_class(f"{args.mod[0]}", new_file, topo, args)
        print("classification done")
        png_paths = get_png_file(f"out_{args.mod[0]}_result")
        png_listing(png_paths)
        print("Choose with which data to proceed (0,1 or 2) : ")
        x = int(input())
        while x not in [0, 1, 2]:
            print("wrong number")
            x = int(input())
        argpaths = get_arg_file(f"out_{args.mod[0]}_result")
        print("Making training and testing set :")
        argvalues = arg_listing(argpaths[x])
        xtrain, xtest = data_making(argvalues)
        print("Autoencoder compiling :")
        autoencoder1 = autoencoder(2, f"{args.mod[0]}")
        autoencoder1.compile(optimizer='adam', loss='mse')
        print("autoencoder compiling done")
        print(autoencoder1.summary())
        print("autoencoder fitting :")
        if args.epoch:
            history = autoencoder1.fit(xtrain, xtrain, epochs=args.epoch[0], batch_size=16, shuffle=True,
                                       validation_data=(xtest, xtest))
        else:
            history = autoencoder1.fit(xtrain, xtrain, epochs=50, batch_size=16, shuffle=True,
                                       validation_data=(xtest, xtest))
        print("done\nplotting...")
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.title(f"{args.mod[0]}/autoencoder loss {x}")
        plt.show()
        print("do you wish to save the encoder?(y/n)")
        choice = input()
        if choice == "y":
            auto = Model(autoencoder1.input, autoencoder1.layers[6].output)
            auto.save(f"{args.mod[0]}_encoder.h5")
            print("model saved")
        print("end")


if __name__ == '__main__':
    main()
