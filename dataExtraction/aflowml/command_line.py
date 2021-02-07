from __future__ import print_function
import argparse
import logging
import json
import os.path
from aflowml import AFLOWmlAPI


def get_prediction():

    logging.basicConfig(format='%(message)s')
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description='Get a prediction for a given POSCAR'
    )

    parser.add_argument(
        'post_data',
        type=str,
        help='Path of the POSCAR file or a composition if model is asc.',
    )

    parser.add_argument(
        '-m',
        '--model',
        required=True,
        help='Specifies the machine learning model to use',
        choices=['plmf', 'mfd', 'asc']
    )

    parser.add_argument(
        '-s',
        '--save',
        action='store_true',
        help='Saves the prediction to a file',
    )

    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Toggle verbose mode',
    )

    parser.add_argument(
        '--outfile',
        type=str,
        help='Specifies the path of the outfile',
    )

    parser.add_argument(
        '--format',
        type=str,
        help='Saves prediction to a file',
        choices=['txt', 'json']
    )

    parser.add_argument(
        '--fields',
        type=str,
        help='Specify the desired fields in the output',
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)

    if os.path.isfile(args.post_data):
        try:
            with open(args.post_data, 'r') as input_file:
                logger.info('reading from %s' % args.post_data)

                ml = AFLOWmlAPI()
                job_id = ml.submit_job(input_file.read(), args.model)

                logger.info('submitted job: %s' % job_id)

                prediction = {}
                if args.fields:
                    fields = [f for f in args.fields.split(',')]
                    prediction = ml.poll_job(job_id, fields=fields)
                else:
                    prediction = ml.poll_job(job_id)

                logger.info('completed job: %s' % job_id)

                if args.save:
                    outfile = ''
                    if args.format is None or args.format == 'txt':
                        outfile = 'prediction.txt'
                        if args.outfile is not None:
                            outfile = args.outfile
                        with open(outfile, 'w') as out:
                            for key, val in prediction.items():
                                out.write('%s = %s \n' % (key, val))

                    elif args.format == 'json':
                        outfile = 'prediction.json'
                        if args.outfile is not None:
                            outfile = args.outfile
                        with open(outfile, 'w') as out:
                            json.dump(prediction, out)
                    logger.info('saved prediction to: %s' % outfile)

                else:
                    logger.info('predicted the following: \n')
                    for key, val in prediction.items():
                        print('%s = %s ' % (key, val))

        except IOError:
            logger.error('ERROR: No such file \'%s\'' % args.post_data)

    elif args.model == 'asc':
        ml = AFLOWmlAPI()
        job_id = ml.submit_job(args.post_data, args.model)

        logger.info('submitted job: %s' % job_id)

        prediction = {}
        if args.fields:
            fields = [f for f in args.fields.split(',')]
            prediction = ml.poll_job(job_id, fields=fields)
        else:
            prediction = ml.poll_job(job_id)

        logger.info('completed job: %s' % job_id)

        if args.save:
            outfile = ''
            if args.format is None or args.format == 'txt':
                outfile = 'prediction.txt'
                if args.outfile is not None:
                    outfile = args.outfile
                with open(outfile, 'w') as out:
                    for key, val in prediction.items():
                        out.write('%s = %s \n' % (key, val))

            elif args.format == 'json':
                outfile = 'prediction.json'
                if args.outfile is not None:
                    outfile = args.outfile
                with open(outfile, 'w') as out:
                    json.dump(prediction, out)
            logger.info('saved prediction to: %s' % outfile)

        else:
            logger.info('predicted the following: \n')
            for key, val in prediction.items():
                print('%s = %s ' % (key, val))
