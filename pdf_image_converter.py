import sys
from time import time
import subprocess
import argparse


def convert(infile, density=300, resize='25%'):
    assert infile.split('.')[-1] == 'pdf'
    outfile = '.'.join(infile.split('.')[:-1] + ['png'])
    s = 'convert -density {density} {input} -resize {resize} {out}'.format(
        input=infile, out=outfile, density=300, resize='25%'
    )
    return subprocess.call(s, shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert from a pdf to a png'
    )
    parser.add_argument('infile', metavar='/path/to/file.pdf', nargs='*',
                        help='PDF image to convert (file path)')
    parser.add_argument('--resize', '-r', metavar='N%', default='25%',
                        help='Output image size')
    parser.add_argument('--density', '-d', metavar='M', default=300,
                        help='output image density (number)')
    args = parser.parse_args()
    t0 = time()
    for i, f in enumerate(args.infile):
        sys.stdout.write('\r-{}> {} / {}'.format('-'*i, i, len(args.infile)))
        sys.stdout.flush()
        convert(f, args.density, args.resize)
    print('\nCompleted in {:.3f}s'.format(time() - t0))
