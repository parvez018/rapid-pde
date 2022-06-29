import sys


def get_cmd_inputs():
    if len(sys.argv)<7:
        print("python %s skip_step batch_size epochs dim comp rid"%(sys.argv[0]))
        sys.exit(1)
    return int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),float(sys.argv[5]),int(sys.argv[6])
