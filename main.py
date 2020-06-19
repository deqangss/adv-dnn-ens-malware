from __future__ import print_function
import sys
from argparse import ArgumentParser

from common import Learner, Attacker, Defender

def _main():
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(title= 'Role selection')
    # create arguments for learner
    learner_parser = subparsers.add_parser('learner', help = 'Learning a basic neural network model.')
    learner_parser.add_argument('-t', '--train', help = 'Training a model',
                                action= 'store_true', default=False)
    learner_parser.add_argument('-p', '--prediction', help = 'Predict labels for test samples.',
                                action= 'store_true', default= False)
    learner_parser.add_argument('-a', '--adversary', help='Predict labels for adversarial malware samples.',
                                 action='store_true', default=False)
    learner_parser.set_defaults(action = 'learner')

    # create arguments for attacker
    attacker_parser = subparsers.add_parser('attacker', help = 'Wage an attack.')
    attacker_parser.add_argument('-s', '--scenario', dest='scenario', type = str, default= 'white-box',
                                 choices=['white-box', 'grey-box', 'black-box'], required=False)
    attacker_parser.add_argument('-v', '--victimmodel', dest='victimmodel', type = str, default= 'basic_dnn',
                                 choices=['basic_dnn',
                                          'atrfgsm',
                                          'atadam',
                                          'atma',
                                          'adema',
                                          'dadema'
                                          ], required= False)
    attacker_parser.add_argument('-m', '--method', dest= 'method', type = str, default= 'fgsm',
                                 choices=['bca_k',
                                          'bga_k',
                                          'fgsm',
                                          'gdkde',
                                          'grosse',
                                          'jsma',
                                          'mimicry',
                                          'pgdlinf',
                                          'pgdl2',
                                          'pgdl1',
                                          'pgd_adam',
                                          'saltandpepper',
                                          'pointwise',
                                          'max'],
                                 required= False)
    attacker_parser.add_argument('-r', '--reality', help='Generate executable adversarial malware samples',
                                 action = 'store_true', default= False)
    attacker_parser.set_defaults(action = 'attacker')

    # create arguments for defender
    defender_parser = subparsers.add_parser('defender', help = 'Defend the classifier (learner).')
    defender_parser.add_argument('-d', '--defense', dest='defense', type = str, default= 'adv_training_dnn',
                                 choices=['atrfgsm', # hardened DNN incorporating adversarial training with r-fgsm
                                          'atadam', # hardened DNN incorporating adversarial training with adam
                                          'atma', # hardened DNN incorporating adversarial training with a mixture of attacks
                                          'adema', # hardened ensemble-based DNN incorporating adversarial training with a mixture of attacks
                                          'dadema' # promoting the diversity of adversarial deep ensemble
                                          ],
                                 required= False)
    defender_parser.add_argument('-t', '--train', help = 'Training defense model', action= 'store_true', default=False)
    defender_parser.add_argument('-p', '--prediction', help = 'Predict labels for test samples.',
                                action= 'store_true', default= False)
    defender_parser.add_argument('-a', '--adversary', help='Predict labels for adversarial malware samples.',
                                 action='store_true', default=False)
    defender_parser.add_argument('-m', '--method', dest= 'method', type = str, default= 'jsma',
                                 choices=['bca_k',
                                          'bga_k',
                                          'fgsm',
                                          'gdkde',
                                          'grosse',
                                          'jsma',
                                          'mimicry',
                                          'pgdlinf',
                                          'pgdl2'
                                          'pgdl1',
                                          'pgd_adam',
                                          'saltandpepper',
                                          'pointwise',
                                          'max'],
                                 required= False)
    defender_parser.set_defaults(action = 'defender')

    args = parser.parse_args()

    if args.action == 'learner':
        learner = Learner()
        if args.train:
            learner.train()
        if args.prediction:
            learner.predict()
        if args.adversary:
            learner.pred_adv()

    elif args.action == 'attacker':
        attacker = Attacker(args.scenario, args.victimmodel, args.method, args.reality)
        sys.exit(attacker.attack())

    # elif args.action == 'oracle':
    #     oracle = Oracle(args.virustotal, args.droidsandbox)
    #     oracle.run()

    elif args.action == 'defender':
        defender = Defender(args.defense)

        if args.train:
            defender.train()
        if args.prediction:
            defender.predict()
        if args.adversary:
            defender.pred_adv()


if __name__ == "__main__":
    sys.exit(_main())