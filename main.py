import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-q',
                        '--question',
                        help="Defines the question number.",
                        default='1.1')
    args = parser.parse_args()
    question = args.question

    if question == '1.1':
        os.system("python fit_data.py --type 'vox'")

    elif question == '1.2':
        os.system("python fit_data.py --type 'point'")

    elif question == '1.3':
        os.system("python fit_data.py --type 'mesh'")

    elif question == '2.1':
        os.system("python train_model.py --type 'vox' --max_iter 10001 --save_freq 2000")
        os.system("python eval_model.py --type 'vox' --load_checkpoint --load_step 10000 --vis_freq 20")

    elif question == '2.2':
        os.system("python train_model.py --type 'point' --max_iter 10001 --save_freq 2000")
        os.system("python eval_model.py --type 'point' --load_checkpoint --load_step 10000 --vis_freq 20")

    elif question == '2.3':
        os.system("python train_model.py --type 'mesh' --max_iter 10001 --save_freq 2000")
        os.system("python eval_model.py --type 'mesh' --load_checkpoint --load_step 10000 --vis_freq 20")

    elif question == '2.6':
        os.system("python interpret_model.py --load_step 10000 --index1 100 --index2 340")

    elif question == '3.1':
        os.system("python train_implicit.py --save_freq 2000 --max_iter 10001")
        os.system("python eval_model.py --type 'implicit' --load_checkpoint --load_step 10000 --vis_freq 20")

    else:
        print('Invalid question!')