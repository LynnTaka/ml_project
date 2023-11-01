from preprocessing import *


if __name__ == '__main__':
    #change path depending on where data is located
    datapath = './alzheimer.csv'

    dp = read_data(datapath)
    X, y = encode_data(dp)
    find_important_features(X, y)
