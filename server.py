import os
import copy
import torch
from fairseq.models.roberta import RobertaModel
from multiprocessing import Pool

cmd_fed_cold = ' fairseq-train $DATA_PATH \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --batch-size $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_base \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --save-interval $SAVE_INTERVAL \
    --max-epoch $MAX_EPOCH \
    --save-dir $SAVE_DIR\
    --find-unused-parameters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric'

cmd_fed_warm = ' fairseq-train $DATA_PATH \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --batch-size $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_base \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --save-interval $SAVE_INTERVAL \
    --max-epoch $MAX_EPOCH \
    --save-dir $SAVE_DIR\
    --find-unused-parameters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric'

cmd_distill = 'CUDA_VISIBLE_DEVICES=1 fairseq-train $DATA_PATH \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --batch-size $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_base \
    --criterion sentence_prediction_distill \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --save-interval $SAVE_INTERVAL \
    --max-epoch $MAX_EPOCH \
    --save-dir $SAVE_DIR\
    --find-unused-parameters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric'

def run_client(fed_no, device_no, epoch_per_fed):
    print("client " + str(device_no) + " starts!")
    GPU_no = device_no % 2 + 1
    # GPU_no = "1"
    # GPU_no = device_no
    max_epoch = (fed_no + 1) * epoch_per_fed
    dir = "client" + str(device_no)
    os.chdir(dir)
    os.system('pwd')
    os.environ['TOTAL_NUM_UPDATES'] = "20935"
    os.environ['WARMUP_UPDATES'] = "1256"
    os.environ['LR'] = "1e-05"
    # os.environ['HEAD_NAME'] = "mnli_head"
    os.environ['NUM_CLASSES'] = "2"
    os.environ['MAX_SENTENCES'] = "32"
    os.environ['DATA_PATH'] = "SST-2-bin/" #modify
    # os.environ['SAVE_INTERVAL'] = str(epoch_per_fed)
    os.environ['SAVE_INTERVAL'] = "1"
    os.environ['SAVE_DIR'] = "checkpoints_sst_2/"
    os.environ['MAX_EPOCH'] = str(max_epoch)
    if fed_no == 0:
        os.environ['ROBERTA_PATH'] = "../roberta.base/model.pt" #modify
        os.system('CUDA_VISIBLE_DEVICES=' + str(GPU_no) + cmd_fed_cold)
    else:
        os.environ['ROBERTA_PATH'] = "checkpoints_sst_2/checkpoint_last.pt" #modify
        os.system('CUDA_VISIBLE_DEVICES=' + str(GPU_no) + cmd_fed_warm)
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(device_no)

def fed_avg():
    roberta_0 = torch.load('client0/checkpoints_sst_2/checkpoint_best.pt', map_location=torch.device("cpu"))
    roberta_1 = torch.load('client1/checkpoints_sst_2/checkpoint_best.pt', map_location=torch.device("cpu"))
    roberta_2 = torch.load('client2/checkpoints_sst_2/checkpoint_best.pt', map_location=torch.device("cpu"))
    roberta_3 = torch.load('client3/checkpoints_sst_2/checkpoint_best.pt', map_location=torch.device("cpu"))

    global_model = copy.deepcopy(roberta_0)
    # global_model_param = copy.deepcopy(roberta_0['model'])

    for layer, param in roberta_1['model'].items():
        # global_model_param[layer] += param
        global_model['model'][layer] += param

    for layer, param in roberta_2['model'].items():
        # global_model_param[layer] += param
        global_model['model'][layer] += param

    for layer, param in roberta_3['model'].items():
        # global_model_param[layer] += param
        global_model['model'][layer] += param

    for layer in global_model['model'].keys():
        # global_model_param[layer] /= 4
        global_model['model'][layer] /= 4
        # print(layer, "\t", global_model_param[layer].size())
    
    torch.save(global_model, 'global_fed.pt')

def distill(fed_no, distill_epochs):
    os.environ['TOTAL_NUM_UPDATES'] = "20935"
    os.environ['WARMUP_UPDATES'] = "1256"
    os.environ['LR'] = "5e-06"
    os.environ['NUM_CLASSES'] = "2"
    os.environ['MAX_SENTENCES'] = "32"
    os.environ['DATA_PATH'] = "distill_data/SST-2-bin/" #modify
    # os.environ['ROBERTA_PATH'] = "./global_fed.pt" #modify
    # os.environ['SAVE_INTERVAL'] = str(distill_epochs)
    os.environ['SAVE_INTERVAL'] = "1"
    os.environ['SAVE_DIR'] = "distill/"
    os.environ['MAX_EPOCH'] = str(distill_epochs)
    # os.environ['MAX_EPOCH'] = "1"
    if fed_no == 0:
        os.environ['ROBERTA_PATH'] = "./roberta.base/model.pt"
    else:
        os.environ['ROBERTA_PATH'] = "distill/checkpoint_best.pt" #modify

    os.system(cmd_distill)

def send_to_client(is_distill):
    if is_distill:
        global_model = torch.load('distill/checkpoint_best.pt', map_location=torch.device("cpu"))
    else:
        global_model = torch.load('global_fed.pt', map_location=torch.device("cpu"))
    
    roberta_0 = torch.load('client0/checkpoints_sst_2/checkpoint_last.pt', map_location=torch.device("cpu"))
    roberta_1 = torch.load('client1/checkpoints_sst_2/checkpoint_last.pt', map_location=torch.device("cpu"))
    roberta_2 = torch.load('client2/checkpoints_sst_2/checkpoint_last.pt', map_location=torch.device("cpu"))
    roberta_3 = torch.load('client3/checkpoints_sst_2/checkpoint_last.pt', map_location=torch.device("cpu"))

    roberta_0['model'] = global_model['model']
    roberta_1['model'] = global_model['model']
    roberta_2['model'] = global_model['model']
    roberta_3['model'] = global_model['model']      
    torch.save(roberta_0, 'client0/checkpoints_sst_2/checkpoint_last.pt')
    torch.save(roberta_1, 'client1/checkpoints_sst_2/checkpoint_last.pt')
    torch.save(roberta_2, 'client2/checkpoints_sst_2/checkpoint_last.pt')
    torch.save(roberta_3, 'client3/checkpoints_sst_2/checkpoint_last.pt')


if __name__=='__main__':
    epoch_total = 5
    epoch_per_fed = 1
    fed_round_num = int(epoch_total / epoch_per_fed)
    is_distill = True

    for i in range(fed_round_num): 
        print("\nfed starts in round " + str(i+1) + " \n")
        p = Pool(4)
        for j in range(4):
            p.apply_async(func=run_client, args=(i, j, epoch_per_fed,))
        p.close()
        p.join()
        print("\nfed ends in round " + str(i+1) + " \n")

        # print("\nfed_avg starts in round " + str(i+1) + " \n")
        # fed_avg()
        # print("\nfed_avg ends in round " + str(i+1) + " \n")
        
        if is_distill:
            distill_epochs = 4
            print("\ndistill starts in round " + str(i+1) + " \n")
            distill(i, distill_epochs)
            print("\ndistill ends in round " + str(i+1) + " \n")

        send_to_client(is_distill)