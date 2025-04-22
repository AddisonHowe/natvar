import sys

def main():
    from natvar.main import parse_args, main
    args = parse_args(sys.argv[1:])
    main(args)

def query_genome():
    from natvar.query_genome import parse_args, main
    args = parse_args(sys.argv[1:])
    main(args)

def query_genome_batch():
    from natvar.query_genome_batch import parse_args, main
    args = parse_args(sys.argv[1:])
    main(args)

def read_genome_sizes():
    from natvar.read_genome_sizes import parse_args, main
    args = parse_args(sys.argv[1:])
    main(args)

if __name__ == "__main__":
    main()
