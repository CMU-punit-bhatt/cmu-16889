import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-q',
                        '--question',
                        help="Defines the question number.",
                        default='1')
    args = parser.parse_args()
    question = args.question

    if question == '1':
        os.system("python train.py")
        os.system("python eval_cls.py --load_checkpoint best_model")

    elif question == '2':
        os.system("python train.py --task seg")
        os.system("python eval_seg.py --load_checkpoint best_model")

    # Exp1 - Perturb
    elif question == '3-cls-exp1':
        os.system("python eval_cls.py --load_checkpoint best_model --exp_name perturb")
        os.system("python eval_cls.py --load_checkpoint best_model --exp_name perturb --perturb_scale 1")
        os.system("python eval_cls.py --load_checkpoint best_model --exp_name perturb --perturb_scale 1.5")
        os.system("python eval_cls.py --load_checkpoint best_model --exp_name perturb --perturb_scale 2")
        os.system("python eval_cls.py --load_checkpoint best_model --exp_name perturb --perturb_scale 3")
        os.system("python eval_cls.py --load_checkpoint best_model --exp_name perturb --perturb_scale 4")
        os.system("python eval_cls.py --load_checkpoint best_model --exp_name perturb --perturb_scale 5")
        os.system("python eval_cls.py --load_checkpoint best_model --exp_name perturb --perturb_scale 6")
        os.system("python eval_cls.py --load_checkpoint best_model --exp_name perturb --perturb_scale 7")
        os.system("python eval_cls.py --load_checkpoint best_model --exp_name perturb --perturb_scale 8")

    # Exp2 - Rotate
    elif question == '3-cls-exp2':
        os.system("python eval_cls.py --load_checkpoint best_model --exp_name rotate --rotation_angle 5")
        os.system("python eval_cls.py --load_checkpoint best_model --exp_name rotate --rotation_angle 10")
        os.system("python eval_cls.py --load_checkpoint best_model --exp_name rotate --rotation_angle 15")
        os.system("python eval_cls.py --load_checkpoint best_model --exp_name rotate --rotation_angle 20")
        os.system("python eval_cls.py --load_checkpoint best_model --exp_name rotate --rotation_angle 30")
        os.system("python eval_cls.py --load_checkpoint best_model --exp_name rotate --rotation_angle 45")
        os.system("python eval_cls.py --load_checkpoint best_model --exp_name rotate --rotation_angle 60")
        os.system("python eval_cls.py --load_checkpoint best_model --exp_name rotate --rotation_angle 90")

    # Exp3 - Num_Points
    elif question == '3-cls-exp3':
        os.system("python eval_cls.py --load_checkpoint best_model --num_points 10")
        os.system("python eval_cls.py --load_checkpoint best_model --num_points 25")
        os.system("python eval_cls.py --load_checkpoint best_model --num_points 50")
        os.system("python eval_cls.py --load_checkpoint best_model --num_points 75")
        os.system("python eval_cls.py --load_checkpoint best_model --num_points 100")
        os.system("python eval_cls.py --load_checkpoint best_model --num_points 500")
        os.system("python eval_cls.py --load_checkpoint best_model --num_points 1000")
        os.system("python eval_cls.py --load_checkpoint best_model --num_points 2000")
        os.system("python eval_cls.py --load_checkpoint best_model --num_points 4000")
        os.system("python eval_cls.py --load_checkpoint best_model --num_points 6000")
        os.system("python eval_cls.py --load_checkpoint best_model --num_points 8000")

    # Exp1 - Perturb
    elif question == '3-seg-exp1':
        os.system("python eval_seg.py --load_checkpoint best_model --exp_name perturb --bad_accuracy 0.5")
        os.system("python eval_seg.py --load_checkpoint best_model --exp_name perturb --perturb_scale 1 --bad_accuracy 0.4")
        os.system("python eval_seg.py --load_checkpoint best_model --exp_name perturb --perturb_scale 1.5 --bad_accuracy 0.3")
        os.system("python eval_seg.py --load_checkpoint best_model --exp_name perturb --perturb_scale 2 --bad_accuracy 0.25")
        os.system("python eval_seg.py --load_checkpoint best_model --exp_name perturb --perturb_scale 3 --bad_accuracy 0.2")
        os.system("python eval_seg.py --load_checkpoint best_model --exp_name perturb --perturb_scale 4 --bad_accuracy 0.2")
        os.system("python eval_seg.py --load_checkpoint best_model --exp_name perturb --perturb_scale 5 --bad_accuracy 0.2")
        os.system("python eval_seg.py --load_checkpoint best_model --exp_name perturb --perturb_scale 6 --bad_accuracy 0.2")
        os.system("python eval_seg.py --load_checkpoint best_model --exp_name perturb --perturb_scale 7 --bad_accuracy 0.2")
        os.system("python eval_seg.py --load_checkpoint best_model --exp_name perturb --perturb_scale 8 --bad_accuracy 0.1")

    # Exp2 - Rotate
    elif question == '3-seg-exp2':
        os.system("python eval_seg.py --load_checkpoint best_model --exp_name rotate --rotation_angle 5 --bad_accuracy 0.5")
        os.system("python eval_seg.py --load_checkpoint best_model --exp_name rotate --rotation_angle 10 --bad_accuracy 0.4")
        os.system("python eval_seg.py --load_checkpoint best_model --exp_name rotate --rotation_angle 15 --bad_accuracy 0.3")
        os.system("python eval_seg.py --load_checkpoint best_model --exp_name rotate --rotation_angle 20 --bad_accuracy 0.25")
        os.system("python eval_seg.py --load_checkpoint best_model --exp_name rotate --rotation_angle 30 --bad_accuracy 0.2")
        os.system("python eval_seg.py --load_checkpoint best_model --exp_name rotate --rotation_angle 45 --bad_accuracy 0.2")
        os.system("python eval_seg.py --load_checkpoint best_model --exp_name rotate --rotation_angle 60 --bad_accuracy 0.2")
        os.system("python eval_seg.py --load_checkpoint best_model --exp_name rotate --rotation_angle 90 --bad_accuracy 0.2")

    # Exp3 - Num_Points
    elif question == '3-seg-exp3':
        os.system("python eval_seg.py --load_checkpoint best_model --num_points 10 --bad_accuracy 0.1")
        os.system("python eval_seg.py --load_checkpoint best_model --num_points 25 --bad_accuracy 0.2")
        os.system("python eval_seg.py --load_checkpoint best_model --num_points 50 --bad_accuracy 0.2")
        os.system("python eval_seg.py --load_checkpoint best_model --num_points 75 --bad_accuracy 0.2")
        os.system("python eval_seg.py --load_checkpoint best_model --num_points 100 --bad_accuracy 0.2")
        os.system("python eval_seg.py --load_checkpoint best_model --num_points 500 --bad_accuracy 0.25")
        os.system("python eval_seg.py --load_checkpoint best_model --num_points 1000 --bad_accuracy 0.25")
        os.system("python eval_seg.py --load_checkpoint best_model --num_points 2000 --bad_accuracy 0.3")
        os.system("python eval_seg.py --load_checkpoint best_model --num_points 4000 --bad_accuracy 0.3")
        os.system("python eval_seg.py --load_checkpoint best_model --num_points 6000 --bad_accuracy 0.3")
        os.system("python eval_seg.py --load_checkpoint best_model --num_points 8000 --bad_accuracy 0.4")

    elif question == '4':
        os.system("python train_locality.py")
        os.system("python eval_cls_locality.py --load_checkpoint best_model")

    else:
        print('Invalid question!')