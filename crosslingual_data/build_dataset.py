import csv
import os
from datasets import load_dataset
import argparse

template_dict = {"en": "This sentence : \"*sent 0*\" means in one word:\"",
                 "ar": "هذه الجملة : \"*sent 0*\" تعني بكلمة واحدة:\"",
                 "ru": "Это предложение : \"*sent 0*\" означает одним словом:\"",
                 "zh": "这句话 ： “*sent 0*” 用一个词来表示是:“",
                 "jp": "この文 ：「*sent 0*」 の意味は一言で：「",
                 "de": "Dieser Satz : \"*sent 0*\" bedeutet in einem Wort:\"",
                 "es": "Esta oración : \"*sent 0*\"  significa en una palabra:\"",
                 "tr": "Bu cümle : \"*send 0*\" tek kelimeyle şu anlama gelir:\"",}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template_type', type=str, default="en-prompts")
    parser.add_argument('--data_type', type=str, default="en2x")
    parser.add_argument('--dataset', type=str, default="NTREX")
    parser.add_argument("--lang_list", type=str, default='en,ar,es')
    parser.add_argument("--data_size", type=int, default=100)
    args = parser.parse_args()

    dataset = args.dataset
    lang_list = args.lang_list.split(',')
    template_type = args.template_type
    data_type = args.data_type
    data_size = int(args.data_size)

    sentences = []

    for src_idx, src_lang in enumerate(lang_list):
        # use EN as source language only
        if src_lang != 'en' and data_type == "en2x":
            continue
        for tgt_idx in range(src_idx+1, len(lang_list)):
            tgt_lang = lang_list[tgt_idx]

            src_file = os.path.join(dataset, src_lang)
            tgt_file = os.path.join(dataset, tgt_lang)

            cnt = 0
            with open(src_file) as src_fin, open(tgt_file) as tgt_fin:
                for src_sent, tgt_sent in zip(src_fin, tgt_fin):
                    fake_neg = "-"
                    if template_type == "self-prompts":
                        src_sent = template_dict[src_lang].replace('*sent 0*', src_sent).strip()
                        tgt_sent = template_dict[tgt_lang].replace('*sent 0*', tgt_sent).strip()
                    elif template_type == "en-prompts":
                        src_sent = template_dict["en"].replace('*sent 0*', src_sent).strip()
                        tgt_sent = template_dict["en"].replace('*sent 0*', tgt_sent).strip()
                    else:
                        src_sent = src_sent
                        tgt_sent = tgt_sent
                    sentences.append((src_sent, tgt_sent, fake_neg))
                    sentences.append((tgt_sent, src_sent, fake_neg))
                    cnt += 1
                    if 0 < data_size <= cnt:
                        break

    data_path = os.path.join(dataset, '{}-{}-{}.csv'.format(dataset, template_type, data_type))

    with open(data_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['sent0', 'sent1', 'hard_neg'])
        csv_writer.writerows(sentences)

if __name__ == "__main__":
    main()

