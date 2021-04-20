import os
import copy
import torch
from fairseq.models.roberta import RobertaModel
from multiprocessing import Pool

cmd0 = ' fairseq-train $DATA_PATH \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --batch-size $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_base \
    --classification-head-name $HEAD_NAME \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --log-format simple --log-interval 1 \
    --save-interval $SAVE_INTERVAL \
    --max-epoch $MAX_EPOCH \
    --find-unused-parameters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric'

cmd1 = ' fairseq-train $DATA_PATH \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --batch-size $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_base \
    --classification-head-name $HEAD_NAME \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --log-format simple --log-interval 1 \
    --save-interval $SAVE_INTERVAL \
    --max-epoch $MAX_EPOCH \
    --find-unused-parameters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric'

def run_client(fed_no, device_no, epoch_per_fed):
    print("cuda " + str(device_no) + " starts!")
    max_epoch = (fed_no + 1) * epoch_per_fed
    dir = "cu" + str(device_no)
    os.chdir(dir)
    os.system('pwd')
    os.environ['TOTAL_NUM_UPDATES'] = "123873"
    os.environ['WARMUP_UPDATES'] = "7432"
    os.environ['LR'] = "1e-05"
    os.environ['HEAD_NAME'] = "mnli_head"
    os.environ['NUM_CLASSES'] = "3"
    os.environ['MAX_SENTENCES'] = "32"
    os.environ['DATA_PATH'] = "MNLI-bin/" #modify
    os.environ['SAVE_INTERVAL'] = str(epoch_per_fed)
    os.environ['MAX_EPOCH'] = str(max_epoch)
    if fed_no == 0:
        os.environ['ROBERTA_PATH'] = "../roberta.base/model.pt" #modify
        os.system('CUDA_VISIBLE_DEVICES=' + str(device_no) + cmd0)
        # os.environ['LR'] = "2e-05"
    else:
        os.environ['ROBERTA_PATH'] = "./checkpoints/checkpoint_last.pt" #modify
        os.system('CUDA_VISIBLE_DEVICES=' + str(device_no) + cmd1)
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(device_no)

def fed_avg():
    roberta_0 = torch.load('cu0/checkpoints/checkpoint_last.pt', map_location=torch.device("cpu"))
    roberta_1 = torch.load('cu1/checkpoints/checkpoint_last.pt', map_location=torch.device("cpu"))
    roberta_2 = torch.load('cu2/checkpoints/checkpoint_last.pt', map_location=torch.device("cpu"))
    roberta_3 = torch.load('cu3/checkpoints/checkpoint_last.pt', map_location=torch.device("cpu"))

    global_model_param = copy.deepcopy(roberta_0['model'])

    for layer, param in roberta_1['model'].items():
        global_model_param[layer] += param

    for layer, param in roberta_2['model'].items():
        global_model_param[layer] += param

    for layer, param in roberta_3['model'].items():
        global_model_param[layer] += param

    for layer in global_model_param.keys():
        global_model_param[layer] /= 4
        # print(layer, "\t", global_model_param[layer].size())
    
    roberta_0['model'] = global_model_param
    roberta_1['model'] = global_model_param
    roberta_2['model'] = global_model_param
    roberta_3['model'] = global_model_param

    torch.save(roberta_0, 'cu0/checkpoints/checkpoint_last.pt')
    torch.save(roberta_1, 'cu1/checkpoints/checkpoint_last.pt')
    torch.save(roberta_2, 'cu2/checkpoints/checkpoint_last.pt')
    torch.save(roberta_3, 'cu3/checkpoints/checkpoint_last.pt')
    
    torch.save(roberta_0, 'checkpoints/checkpoint_last.pt')
   
if __name__=='__main__':
    epoch_total = 10
    epoch_per_fed = 5
    fed_num = int(epoch_total / epoch_per_fed)     
    for i in range(fed_num): 
        print("\nfed " + str(i+1) + " starts\n")
        p = Pool(4)
        for j in range(4):
            p.apply_async(func=run_client, args=(i, j, epoch_per_fed,))
        p.close()
        p.join()
        print("\nfed " + str(i+1) + " ends\n")
        fed_avg()