
from mlfire.utils.functool import lazy_import

# lazy imports
argparse = lazy_import('argparse')


def cli_argument_parser() -> dict:

    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--img_dir',
                        metavar='DATA_DIR',
                        type=str,
                        default='data/tifs',
                        required=False)

    args = parser.parse_args()

    kwargs = {
        'img_dir': args.img_dir
    }

    return kwargs
