import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cora", help="Cora, CiteSeer, PubMed, Computers, Photo")

    # masking
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument('--imb_ratio', type=float, default=10,help='Imbalance Ratio')

    # Encoder
    parser.add_argument("--dim", type=int, default=128, help="64,128,256")
    parser.add_argument("--layers", type=int, default=2, help="1,2,3")
    parser.add_argument('--n_head', type=int, default=8,help='the number of heads in GAT')
    parser.add_argument('--net', type=str, default='GCN',help='GCN,GAT,SAGE,SGC,PPNP,CHEB')
    parser.add_argument('--chebgcn_para', type=int, default=2, help=' Chebyshev filter size of ChebConv')


    # optimization
    parser.add_argument("--epochs", '-e', type=int, default=2000, help="The number of epochs")
    parser.add_argument("--lr", '-lr', type=float, default=0.005, help="Learning rate. Default is 0.0001.")
    parser.add_argument("--decay", type=float, default=5e-4, help="Learning rate. Default is 0.0001.")
    parser.add_argument("--patience", type=int, default=300)


    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--clustering", action='store_true', default=True)
    parser.add_argument("--num_K", type=int, default=200)
    parser.add_argument("--stride", '-s', type=float, default=1.0, help="stride of round")
    parser.add_argument("--threshold", type=float, default=0.25, help="distance threshold")
    parser.add_argument("--rbo", type=float, default=0.5, help="rbo weight")
    parser.add_argument("--ad", type=float, default=4, help="adding nodes of each round")


    return parser.parse_known_args()[0]
