import gc
import argparse
from metric_tools.style_transfer_accuracy import *
from metric_tools.content_similarity import *
from metric_tools.fluency import *
from metric_tools.joint_metrics import *
import numpy as np
import pandas as pd

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", help="input file", required=True)

    parser.add_argument("--cola_classifier_path", 
                       default='./evaluation_detox/cola_classifier'
                       )
    parser.add_argument("--wieting_model_path",
                        default='./evaluation_detox/sim.pt'
                        )
    parser.add_argument("--wieting_tokenizer_path",
                        default='./evaluation_detox/sim.sp.30k.model'
                        )

    parser.add_argument("--batch_size", default=32, type=int)
    
    parser.add_argument("--t1", default=75, type=float)
    parser.add_argument("--t2", default=70, type=float)
    parser.add_argument("--t3", default=12, type=float)
    
    parser.add_argument("--toxification", action='store_true')
    parser.add_argument("--task_name", choices=['jigsaw', 'yelp'], default='jigsaw')
    args = parser.parse_args()


    # with open(args.inputs, 'r') as input_file, open(args.preds, 'r') as preds_file:
    #     inputs = input_file.readlines()
    #     preds = preds_file.readlines()
    inputfile = args.input
    df = pd.read_csv(inputfile)  # 替换为你的文件路径
    inputs = df["toxic"].tolist()
    preds = df["non-toxic"].tolist()

    # accuracy of style transfer
    #STA
    accuracy_by_sent = classify_preds(args, preds)
    accuracy = np.mean(accuracy_by_sent)
    print("accuracy:",accuracy)
    cleanup()
    
    # similarity
    # bleu = calc_bleu(inputs, preds)

    emb_sim_stats = flair_sim(args, inputs, preds)
    emb_sim = emb_sim_stats.mean()
    print("emb_sim:",emb_sim)
    cleanup()

    # SIM
    similarity_by_sent = wieting_sim(args, inputs, preds)
    avg_sim_by_sent = similarity_by_sent.mean()
    print("avg_sim_by_sent:",avg_sim_by_sent)
    cleanup()
    
    # fluency
    # char_ppl = calc_flair_ppl(preds)
    # char_ppl = 0
    # cleanup()
    
    # token_ppl = calc_gpt_ppl(preds)
    # print("token_ppl:",token_ppl)
    # cleanup()
    
    cola_stats = do_cola_eval(args, preds)
    cola_acc = sum(cola_stats) / len(preds)
    print("cola_acc:",cola_acc)
    cleanup()
 
    # count metrics
    # gm = get_gm(args, accuracy, emb_sim, char_ppl)
    joint = get_j(args, accuracy_by_sent, similarity_by_sent, cola_stats, preds)
    print("joint:",joint)
    
    # write res to table
    if not os.path.exists('./metric_results.md'):
        with open('detoxllm/metric_results.md', 'w') as f:
            f.writelines('| Model | ACC | EMB_SIM | SIM | FL | J | \n')
            f.writelines('| ----- | --- | ------- | --- | -- | - | \n')
            
    with open('./metric_results.md', 'a') as res_file:
        name = args.input
        res_file.writelines(f'{name}|{accuracy:.4f}|{emb_sim:.4f}|{avg_sim_by_sent:.4f}|{cola_acc:.4f}|{joint:.4f}|\n')