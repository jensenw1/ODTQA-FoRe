import argparse
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer
from utils import *
from model import *

# 自定义的导入
# from your_modules import BertClassifier, json2dataframe, train, evaluate

def parse_args():
    parser = argparse.ArgumentParser(description='Training and evaluation of the BERT classifier.')
    # 路径参数
    parser.add_argument('--bert_path', type=str, default='your_path_to/bert-base-chinese', help='Path to the pre-trained BERT model.')
    parser.add_argument('--root_path', type=str, default='../../datasets/', help='Root directory of the QA dataset JSON files.')
    parser.add_argument('--model_save_path', type=str, default='./bert.pt', help='Model saving path.')
    parser.add_argument('--log_dir', type=str, default='./runs', help='TensorBoard log directory.')
    parser.add_argument('--task', type=str, default='train', help='Task category.')
    # 训练参数
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    return parser.parse_args()

def main():
    args = parse_args()
    all_intents = ["Numerical", "Judgment"]
    intents_num = {"Numerical": 0, "Judgment": 1}
    # Initialize tokenizer.
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.task == 'train':
        # Load dataset.
        train_path = args.root_path + 'train.json'
        with open(train_path, 'r', encoding='utf-8') as f:
            all_datas = json.load(f)
        df_train = json2dataframe(all_datas, tokenizer, intents_num)
        validation_path = args.root_path + 'validation.json'
        with open(validation_path, 'r', encoding='utf-8') as f:
            all_datas = json.load(f)
        df_val = json2dataframe(all_datas, tokenizer, intents_num)
        print(f"Train Num:{len(df_train)}, Validation Num:{len(df_val)}")
        # 初始化TensorBoard和模型
        writer = SummaryWriter(args.log_dir)
        model = BertClassifier()
        # 训练和评估
        train(model, df_train, df_val, args.lr, args.epochs, writer, tokenizer, args.batch_size)
        # 保存模型
        torch.save(model, args.model_save_path)
        print(f"The BERT fine-tuned parameters have been saved to: {args.model_save_path}")

    elif args.task == 'test':
        # Load model.
        model_path = args.model_save_path 
        model = torch.load(model_path)
        model.eval()
        test_path = args.root_path + 'test.json'
        with open(test_path, 'r', encoding='utf-8') as f:
            test_datas = json.load(f)
        #df_test = json2dataframe(all_datas, tokenizer, slots_num, intents_num)
        print(f"Test Num:{len(test_datas)}")
        datas = evaluate(model, test_datas, all_intents, tokenizer, device)
        save_json_file = './results/test-with-BERT_pred_query_type.json'
        with open(save_json_file, 'w', encoding='utf-8') as file:
            json.dump(datas, file, ensure_ascii=False, indent=4)
        print(f'For each Query, the corresponding Query Type predicted by BERT have been matched, and the file is saved at: {save_json_file }')


if __name__ == '__main__':
    main()
